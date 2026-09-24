# ssl_aasist.py
#
# SSL-AASIST: a pretrained speech model (wav2vec 2.0 XLS-R, 300M parameters)
# in front of AASIST's graph back end, fine-tuned end to end.
#
# Tak, Todisco, Wang, Jung, Yamagishi, Evans. "Automatic speaker verification
# spoofing and deepfake detection using wav2vec 2.0 and data augmentation."
# Odyssey 2022. Ported from model.py in
# https://github.com/TakHemlata/SSL_Anti-spoofing (MIT licence, the same terms
# and the same upstream lineage as aasist.py — see that file's header).
#
# The pretrained model is facebook/wav2vec2-xls-r-300m (Apache-2.0), pinned to
# XLSR_REVISION below: 436,000 hours of unlabelled speech in 128 languages.
#
# ---------------------------------------------------------------------------
#
# WHY IT IS HERE. RESULTS.md Findings 6 and 7: every model trained here falls
# to 37–49% EER on In-the-Wild, and neither augmentation (RawBoost) nor more
# ASVspoof-style lab data (ASVspoof 5) moved it. Both of those change what the
# model is trained ON. This changes what it knows BEFORE training: XLS-R has
# already learned what real speech sounds like across hundreds of thousands of
# hours, speakers, languages and recording channels, and AASIST only has to
# learn what deviates from it. The small models here learn "real" from 2,580
# VCTK clips.
#
# The front end reads the same standardised 4-second waveform every other
# "raw" model does — preprocess_waveform is unchanged. XLS-R's own feature
# extractor applies exactly that normalisation (zero mean, unit variance per
# utterance), so the input matches what it was pretrained on.
#
# ---------------------------------------------------------------------------
#
# DEVIATIONS FROM UPSTREAM. Read these before comparing to published numbers.
#
# 1. Hugging Face transformers instead of fairseq. Same XLS-R 300M weights
#    (fairseq's xlsr2_300m.pt, converted by Meta and published on the Hub).
#    fairseq does not install cleanly against current torch, and this is the
#    only thing it would be needed for.
#
# 2. Time masking and LayerDrop are OFF. Upstream calls fairseq with
#    mask=False, i.e. no SpecAugment-style masking while fine-tuning. The Hub
#    config ships mask_time_prob=0.075 and layerdrop=0.1 (settings for its own
#    ASR fine-tuning), which transformers would silently apply in train mode.
#    Both are zeroed so the network matches upstream's, and so its depth is
#    the same at train and test time.
#
# 3. Every BatchNorm is a GroupNorm, as in aasist.py (its deviation 1) — the
#    three in upstream's head: first_bn, first_bn1, and the one inside the
#    attention block. XLS-R itself has none (feat_extract_norm="layer").
#    Unlike AASIST this model is NOT expected to train under DP: 300M
#    per-sample gradients are far beyond what DP-SGD can afford, and APPROACH.md
#    already says SSL front-ends come only after dropping DP. GroupNorm is kept
#    anyway so "no BatchNorm anywhere" stays true of every architecture.
#
# 4. aasist.py's deviations 2, 4, 5, 6, 8 and 9 apply unchanged, because the
#    graph back end is aasist.GraphBackend itself, not a copy.
#
# 5. The positional encoding is xavier-initialised like AASIST's rather than
#    torch.randn. Initialisation of 42 x 64 learned values, nothing else.
#
# 6. Input is 64000 samples (this repo's MAX_LEN) rather than 64600, which is
#    199 XLS-R frames against 201. Attention does not care how many nodes it
#    is given.

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aasist import GraphBackend, ResidualBlock, group_norm

XLSR_ID = "facebook/wav2vec2-xls-r-300m"
# The Hub commit the weights are read from. A repo name alone is a moving
# target; a checkpoint trained against one revision should be reproducible.
XLSR_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
# That revision's config.json, committed beside this file, so a network can be
# BUILT with no network access — app.py and evaluate.py build it and then load
# a full checkpoint over it; only the trainer needs the pretrained weights.
XLSR_CONFIG_FILE = Path(__file__).resolve().parent / "xlsr_300m_config.json"

# Upstream's hard-coded values in Model.__init__.
SSL_AASIST_CONFIG = {
    "filts": [128, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.5, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def xlsr_config():
    """The pinned XLS-R config with deviation 2 applied."""
    from transformers import Wav2Vec2Config

    cfg = Wav2Vec2Config(**json.loads(XLSR_CONFIG_FILE.read_text()))
    cfg.mask_time_prob = 0.0
    cfg.mask_feature_prob = 0.0
    cfg.layerdrop = 0.0
    return cfg


class SSLAASIST(GraphBackend):
    """XLS-R -> Linear(1024, 128) -> AASIST encoder and graph.

    Input (batch, 1, samples), output (batch, 2) logits — the same contract as
    every other network here, so the trainer, evaluate.py and app.py need no
    special case.

      waveform -> XLS-R             (batch, 199 frames, 1024)
               -> Linear            (batch, 199, 128): 128 "bands"
               -> max-pool 3x3      (batch, 1, 42, 66)
               -> 6 ResidualBlocks  no time pooling: (batch, 64, 42, 66)
               -> attention pooling 42 spectral nodes, 66 temporal nodes
               -> AASIST's graph    exactly as in aasist.py
    """

    def __init__(self, pretrained=False, config=None):
        """`pretrained=True` downloads XLS-R's weights (1.27 GB, cached under
        $HF_HOME). Only the trainer asks for that. Everything else builds the
        architecture from the committed config and loads a checkpoint over it,
        which already holds the fine-tuned XLS-R."""
        super().__init__()
        from transformers import Wav2Vec2Model

        config = dict(SSL_AASIST_CONFIG if config is None else config)
        filts = config["filts"]
        self.config = config

        if pretrained:
            self.ssl = Wav2Vec2Model.from_pretrained(
                XLSR_ID, revision=XLSR_REVISION, config=xlsr_config())
        else:
            self.ssl = Wav2Vec2Model(xlsr_config())
        self.LL = nn.Linear(self.ssl.config.hidden_size, filts[0])

        self.first_norm = group_norm(1)
        self.first_norm1 = group_norm(filts[-1][-1])

        # Upstream's Residual_block has no time pooling — the SSL frames are
        # already 20 ms apart, against the sinc filterbank's one per sample.
        self.encoder = nn.Sequential(
            ResidualBlock(nb_filts=filts[1], first=True, pool=False),
            ResidualBlock(nb_filts=filts[2], pool=False),
            ResidualBlock(nb_filts=filts[3], pool=False),
            ResidualBlock(nb_filts=filts[4], pool=False),
            ResidualBlock(nb_filts=filts[4], pool=False),
            ResidualBlock(nb_filts=filts[4], pool=False),
        )

        # Where AASIST summarises the encoder output with max(|x|), upstream
        # SSL-AASIST learns a soft attention over each axis instead.
        c = filts[-1][-1]
        self.attention = nn.Sequential(
            nn.Conv2d(c, 128, kernel_size=(1, 1)),
            nn.SELU(),
            group_norm(128),
            nn.Conv2d(128, c, kernel_size=(1, 1)),
        )

        # 42 = the 128 projected bands max-pooled by 3.
        self._build_graph(filts, config["gat_dims"], config["pool_ratios"],
                          config["temperatures"], n_spectral_nodes=filts[0] // 3)

    def forward(self, x):
        """x: (batch, 1, samples) -> (batch, 2) logits."""
        feats = self.ssl(x.squeeze(1)).last_hidden_state   # (batch, frames, 1024)
        x = self.LL(feats).transpose(1, 2).unsqueeze(1)    # (batch, 1, 128, frames)
        x = F.max_pool2d(x, (3, 3))                         # (batch, 1, 42, frames/3)
        x = self.selu(self.first_norm(x))

        x = self.encoder(x)                                 # (batch, 64, 42, T)
        x = self.selu(self.first_norm1(x))

        w = self.attention(x)
        # Spectral nodes: each band, a learned weighted sum over time.
        e_S = torch.sum(x * F.softmax(w, dim=-1), dim=-1)   # (batch, 64, 42)
        # Temporal nodes: each frame, a learned weighted sum over bands.
        e_T = torch.sum(x * F.softmax(w, dim=-2), dim=-2)   # (batch, 64, T)
        return self._graph_readout(e_S.transpose(1, 2), e_T.transpose(1, 2))
