# aasist.py
#
# AASIST — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
# Networks. Jung, Heo, Tak, Shim, Chung, Lee, Yu, Evans, ICASSP 2022.
#
# Ported from the official implementation, models/AASIST.py in
# https://github.com/clovaai/aasist, which carries:
#
#     AASIST
#     Copyright (c) 2021-present NAVER Corp.
#     MIT license
#
#     Permission is hereby granted, free of charge, to any person obtaining a
#     copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to permit
#     persons to whom the Software is furnished to do so, subject to the
#     following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ---------------------------------------------------------------------------
#
# WHAT THIS MODEL DOES, in the terms the rest of this repo uses:
#
# Our CNN reads a spectrogram — a picture of the sound that we chose how to
# draw. RESULTS.md Finding 2 measured the cost of choosing: log-Mel and LFCC
# each win decisively on attacks the other misses, which is only possible if
# both are discarding evidence. AASIST does not take a picture. It reads the
# raw waveform through a bank of band-pass filters whose cutoff frequencies are
# themselves learned (SincNet), so the model decides which frequencies matter.
#
# It then treats the clip as a graph. Some nodes are frequency bands, some are
# time segments, and every node can attend to every other one. A CNN pools,
# i.e. averages, so an artefact confined to one narrow band during one
# half-second is averaged into nothing. Attention can keep it. Handling
# frequency and time in ONE heterogeneous graph, rather than two separate ones,
# is AASIST's contribution over its predecessor RawGAT-ST.
#
# ---------------------------------------------------------------------------
#
# SEVEN DEVIATIONS FROM UPSTREAM. Each is deliberate; none changes the shape of
# the network. Anyone comparing this file to models/AASIST.py should read these
# first, and anyone quoting the published 0.83% EER beside our number should
# know that (1) is a real architectural difference and the rest are not.
#
# 1. EVERY BatchNorm IS A GroupNorm. This is the one that matters and the one
#    the project requires. BatchNorm computes its statistics across the batch,
#    so one sample's normalised value depends on the other samples beside it.
#    That destroys DP-SGD's per-sample gradient guarantee, and Opacus refuses
#    to wrap such a model at all. GroupNorm normalises within each sample and
#    is the standard substitution. APPROACH.md, "Differential privacy: off now,
#    ready later": DP is off for the main results, but every architecture stays
#    DP-compatible, and no BatchNorm anywhere is the whole cost of that.
#
#    This is not free. Published AASIST numbers were produced with BatchNorm,
#    so our result is not a reproduction of theirs even before training
#    differences are counted.
#
# 2. ResidualBlock's `bn1` is GONE, and this changes nothing. Upstream computes
#    `out = self.bn1(x); out = self.selu(out)` and then immediately overwrites
#    `out` with `self.conv1(x)`. The pre-activation result is discarded — the
#    parameters are dead, receive no gradient, and are numerically irrelevant.
#    Removing the module is exactly equivalent to keeping it. It is removed
#    rather than kept because Opacus errors on a trainable parameter that never
#    gets a gradient, so dead weights are not harmless here.
#
# 3. The two `master.expand(...)` lines in forward are GONE, same reason. The
#    expanded tensors are overwritten by the layer call on the next line, which
#    is handed `self.master1` / `self.master2` directly and broadcast instead.
#
# 4. RETURNS LOGITS ONLY. Upstream returns `(last_hidden, output)`. Everything
#    in this repo — the training loop, evaluate.py, app.py — expects a single
#    (batch, 2) tensor, and returning the embedding too would mean every call
#    site handles a tuple for the benefit of one architecture.
#
# 5. INPUT IS (batch, 1, samples), not (batch, samples). That is what
#    preprocess_waveform in model.py already produces for every front-end: a
#    channel axis, then the features. Upstream's `x.unsqueeze(1)` is therefore
#    dropped rather than the caller being made to squeeze.
#
# 6. NO inplace=True on SELU or Dropout. Upstream sets it throughout to save
#    memory. In-place activations mutate tensors that Opacus's backward hooks
#    read, which is a class of bug that is silent rather than loud. The memory
#    saved is not worth it at 297k parameters.
#
# 7. The sinc filterbank is a registered buffer, not a bare attribute, so
#    `model.to(device)` moves it. Upstream re-copies it to the input's device
#    on every forward pass instead. It is non-persistent — it is derived
#    deterministically from the config, so writing it into every checkpoint
#    would be storing a constant.
#
# 8. NO FREE PARAMETERS. Upstream holds the attention weights, the positional
#    encoding and the two master nodes as bare nn.Parameters used through
#    torch.matmul and broadcasting. Here each is the standard module that
#    already computes exactly that: an attention weight is a matmul against an
#    (out_dim, 1) matrix, i.e. nn.Linear(out_dim, 1, bias=False); a positional
#    encoding and a master node are learned vectors looked up by a constant
#    index, i.e. nn.Embedding. Same parameter count, same arithmetic, same
#    initialisation — verified against upstream to float32 rounding.
#
#    This is not tidying. Opacus computes per-sample gradients by hooking
#    MODULES, so a parameter that belongs to no module it recognises never gets
#    one, and a free parameter on the root module also drags the root into
#    Opacus's "trainable layer with buffers" check via the sinc filterbank.
#    With the parameters as written here, AASIST passes make_private. Without
#    them the GroupNorm work in (1) buys nothing, because the model still
#    cannot be trained privately — which is the measurement this project
#    exists to make.
#
# 9. Each ResidualBlock is a direct child of the encoder Sequential rather than
#    being wrapped in its own single-element Sequential. Upstream's extra
#    nesting only changes state_dict key names (encoder.1.0.conv1 becomes
#    encoder.1.conv1), and matters solely to anyone loading upstream weights.

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Official hyperparameters, from config/AASIST.conf in the upstream repo. Not
# retuned: the point of this row in APPROACH.md's comparison table is AASIST as
# published, against our CNN, on the same data.
AASIST_CONFIG = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# config/AASIST-L.conf. 85k parameters against AASIST's 297k — a third the size
# of our current CNN — and still beats everything in the table except full
# AASIST. This is the fallback if anything runs out of memory.
AASIST_L_CONFIG = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def group_norm(channels, channels_per_group=16):
    """A GroupNorm sized for `channels`, standing in for BatchNorm.

    Deviation 1 in the header. num_groups must divide num_channels, and the
    channel counts here (1, 24, 32, 64) do not share a single divisor, so the
    group count is derived rather than fixed: aim for ~16 channels per group,
    then walk down until it divides evenly. Both extremes are valid — one group
    is LayerNorm, one channel per group is InstanceNorm — so this cannot fail.
    """
    groups = max(1, channels // channels_per_group)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _apply_norm(norm, x):
    """Normalise node features shaped (batch, node, dim).

    Upstream flattens to (batch * node, dim) and applies BatchNorm1d, which
    mixes statistics across the batch. GroupNorm wants features on the channel
    axis, so the tensor is transposed and the nodes become the spatial axis:
    each sample is normalised across its own nodes and nothing crosses the
    batch boundary.
    """
    return norm(x.transpose(1, 2)).transpose(1, 2)


def attention_weight(out_dim):
    """The per-edge-type attention weight: upstream's Parameter(out_dim, 1).

    `torch.matmul(x, W)` with W of shape (out_dim, 1) is what a bias-free
    Linear(out_dim, 1) computes, transposed. Deviation 8: as a module, Opacus
    can hook it. Xavier is upstream's initialisation and is unaffected by the
    transpose — it depends on fan_in + fan_out, which is the same either way.
    """
    layer = nn.Linear(out_dim, 1, bias=False)
    nn.init.xavier_normal_(layer.weight)
    return layer


def learned_vectors(count, dim):
    """`count` learned vectors of width `dim`, looked up by constant index.

    Upstream's Parameter(1, count, dim), broadcast over the batch. Used for the
    spectral positional encoding (23 vectors, one per frequency node) and for
    the master nodes (one each). Deviation 8 again.
    """
    table = nn.Embedding(count, dim)
    nn.init.xavier_normal_(table.weight)
    return table


def _lookup(table, batch, count, device):
    """Read every row of `table`, batch-first: (batch, count, dim).

    The indices are constant — this is a broadcast written as a gather — but
    they carry the batch dimension, which is what lets Opacus attribute the
    gradient to the sample it came from.
    """
    idx = torch.arange(count, device=device).expand(batch, count)
    return table(idx)


class GraphAttentionLayer(nn.Module):
    """One homogeneous graph attention layer: all nodes are the same kind.

    Used twice — once over the frequency nodes, once over the time nodes —
    before the two kinds are ever mixed.
    """

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = attention_weight(out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.norm = group_norm(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU()
        self.temp = temperature

    def forward(self, x):
        """x: (batch, node, dim) -> (batch, node, out_dim)"""
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = _apply_norm(self.norm, x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        """Every node times every other node: (batch, node, node, dim)."""
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x):
        """How much each node should listen to each other node: (batch, node, node, 1)."""
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = self.att_weight(att_map)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2


class HtrgGraphAttentionLayer(nn.Module):
    """The heterogeneous layer — AASIST's actual contribution.

    Frequency nodes and time nodes live in one graph, and the four kinds of
    edge get their own attention weights: frequency-to-frequency (11),
    time-to-time (22), and the two crossing directions (12, shared). A `master`
    node sits above all of them, attends to everything, and is attended to by
    nothing — it is a summary of the whole clip, and it is one of the five
    vectors the classifier finally reads.
    """

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = attention_weight(out_dim)
        self.att_weight22 = attention_weight(out_dim)
        self.att_weight12 = attention_weight(out_dim)
        self.att_weightM = attention_weight(out_dim)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.norm = group_norm(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU()
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        """x1, x2: (batch, node, dim) for the two node types. Returns both, plus the master."""
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = _apply_norm(self.norm, x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = self.att_weightM(att_map)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        """Four edge types, four weight vectors, assembled into one attention map."""
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = self.att_weight11(
            att_map[:, :num_type1, :num_type1, :])
        att_board[:, num_type1:, num_type1:, :] = self.att_weight22(
            att_map[:, num_type1:, num_type1:, :])
        att_board[:, :num_type1, num_type1:, :] = self.att_weight12(
            att_map[:, :num_type1, num_type1:, :])
        att_board[:, num_type1:, :num_type1, :] = self.att_weight12(
            att_map[:, num_type1:, :num_type1, :])

        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(
            torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2


class GraphPool(nn.Module):
    """Keep the top k fraction of nodes by learned importance, drop the rest.

    The graph analogue of max pooling: rather than averaging neighbours
    together, it scores every node and discards the least interesting ones, so
    a narrow artefact survives instead of being averaged away.
    """

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        scores = self.sigmoid(self.proj(self.drop(h)))
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class SincConv(nn.Module):
    """The learnable filterbank — `CONV` upstream, SincNet in the literature.

    A bank of `out_channels` band-pass filters, each one a difference of two
    sinc functions windowed by a Hamming window, laid out on the Mel scale at
    initialisation. This is the piece that replaces the spectrogram: instead of
    us fixing how the audio becomes a time-frequency representation, the
    filters are convolution weights and training moves them.

    Note what is NOT thrown away here. A magnitude spectrogram discards phase,
    and vocoders reconstruct phase artificially — so that is evidence deleted
    before the model ever sees it. A convolution over the raw waveform keeps it.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                f"SincConv only supports one input channel (got {in_channels})")
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        # Forced odd so every filter is perfectly symmetric — an even-length
        # sinc has no centre tap and introduces a half-sample delay.
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        nfft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self.to_mel(f)
        # Equal spacing on the Mel scale is the INITIALISATION only. The filters
        # are free to move off it during training, which is the whole argument
        # for this front-end over a fixed Mel filterbank.
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), out_channels + 1)
        self.mel = self.to_hz(filbandwidthsmel)

        hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                             (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(out_channels, self.kernel_size)
        window = torch.from_numpy(np.hamming(self.kernel_size)).float()
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            h_high = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * hsupp.numpy() / self.sample_rate)
            h_low = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * hsupp.numpy() / self.sample_rate)
            band_pass[i, :] = window * torch.from_numpy(h_high - h_low).float()

        # Deviation 7: a buffer, so .to(device) carries it. Non-persistent
        # because it is derived from the config, so a checkpoint storing it
        # would be storing a constant.
        self.register_buffer("band_pass", band_pass, persistent=False)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone()
        if mask:
            # Frequency augmentation: blank a random run of filters, so the
            # model cannot lean on any one band. Off by default — see the
            # freq_aug note on AASIST.forward.
            width = int(np.random.uniform(0, 20))
            start = np.random.randint(0, band_pass_filter.shape[0] - width + 1)
            band_pass_filter[start:start + width, :] = 0

        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, bias=None, groups=1)


class ResidualBlock(nn.Module):
    """`Residual_block` upstream. Two convolutions plus a skip connection.

    The skip is what lets six of these stack without the gradient vanishing:
    each block learns a correction to its input rather than a replacement for
    it. MaxPool2d((1, 3)) shortens time only — the 23 frequency bands are
    preserved all the way to the graph, because a band IS a node there.
    """

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        # Deviation 2: no bn1. Upstream computes it and discards the result.
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU()
        self.norm2 = group_norm(nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(0, 1), stride=1)

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0], out_channels=nb_filts[1],
                padding=(0, 1), kernel_size=(1, 3), stride=1)

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


class AASIST(nn.Module):
    """The full network. `Model` upstream.

    Input is (batch, 1, samples) — the raw waveform, exactly as
    preprocess_waveform produces it under the "raw" front-end. Output is
    (batch, 2) logits, in this repo's fixed label order: bonafide=0, spoof=1.

    The path through it:

      waveform -> SincConv          learnable band-pass filters
               -> |.|, max-pool     magnitude, downsampled 3x in both axes
               -> 6 ResidualBlocks  a (batch, 64, 23, time) feature map
               -> two GATs          23 frequency nodes, ~29 time nodes
               -> two HtrgGAT paths both node types in one graph, twice, and
                                    the two readings maxed together
               -> 5 vectors         time max/avg, freq max/avg, master node
               -> Linear -> 2 logits

    Length independence: the 23 frequency nodes come from 70 sinc filters
    max-pooled by 3, so they do not depend on clip length. The time nodes do,
    and attention does not care how many nodes it is given. At this repo's
    MAX_LEN of 64000 samples the time axis reduces to 29 nodes — the same 29
    the official 64600-sample setting produces, so the 600-sample difference
    costs us nothing.
    """

    def __init__(self, config=None):
        super().__init__()
        config = dict(AASIST_CONFIG if config is None else config)
        filts = config["filts"]
        gat_dims = config["gat_dims"]
        pool_ratios = config["pool_ratios"]
        temperatures = config["temperatures"]
        self.config = config

        self.conv_time = SincConv(out_channels=filts[0],
                                  kernel_size=config["first_conv"],
                                  in_channels=1)
        self.first_norm = group_norm(1)

        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.selu = nn.SELU()

        self.encoder = nn.Sequential(
            ResidualBlock(nb_filts=filts[1], first=True),
            ResidualBlock(nb_filts=filts[2]),
            ResidualBlock(nb_filts=filts[3]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4]),
        )

        # 23 is the frequency-node count out of the encoder, fixed by the 70
        # sinc filters and the 3x pool. A positional encoding is needed because
        # graph attention is permutation-invariant — without it the model could
        # not tell a low band from a high one.
        self.n_spectral_nodes = 23
        self.pos_S = learned_vectors(self.n_spectral_nodes, filts[-1][-1])
        self.master1 = learned_vectors(1, gat_dims[0])
        self.master2 = learned_vectors(1, gat_dims[0])

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, freq_aug=False):
        """x: (batch, 1, samples) -> (batch, 2) logits.

        `freq_aug` blanks a random run of sinc filters per batch, upstream's
        frequency augmentation. The training loop does not pass it — the first
        AASIST run is the plain architecture, so its number is attributable to
        the architecture rather than to an augmentation the CNN rows never had.
        """
        x = self.conv_time(x, mask=freq_aug)       # (batch, 70, samples')
        x = x.unsqueeze(dim=1)                      # (batch, 1, 70, samples')
        x = F.max_pool2d(torch.abs(x), (3, 3))      # magnitude, 3x down both axes
        x = self.first_norm(x)
        x = self.selu(x)

        e = self.encoder(x)                         # (batch, 64, 23, time)

        batch, device = x.size(0), x.device

        # Spectral nodes: one per frequency band, summarised over time.
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + _lookup(
            self.pos_S, batch, self.n_spectral_nodes, device)
        out_S = self.pool_S(self.GAT_layer_S(e_S))

        # Temporal nodes: one per time segment, summarised over frequency.
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        # Two independent readings of the same heterogeneous graph, each with
        # its own master node, maxed together at the end.
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=_lookup(self.master1, batch, 1, device))
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=_lookup(self.master2, batch, 1, device))
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        # Deviation 4: the embedding is dropped, only the logits are returned.
        return self.out_layer(self.drop(last_hidden))


class AASISTLight(AASIST):
    """AASIST-L, the 85k-parameter variant. Same code, smaller config."""

    def __init__(self):
        super().__init__(config=AASIST_L_CONFIG)
