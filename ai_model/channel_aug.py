# channel_aug.py — training-time augmentation: a random, realistic recording
# channel, applied to real and fake clips alike (RESULTS.md Finding 16).
#
# Why this exists. The served model scores clean audio, real or fake, about 14
# log-odds lower than noisy real-world speech (Finding 13). Its training data
# explains how: every clip in LA and SpeechFake, real and fake, is clean
# studio or audiobook audio, so the model never saw a noisy clip of either
# class and has no reason to separate "noisy" from "real". Findings 10-11 added
# noisy real speech and the model learned those sources' recording conditions
# as a sign of realness instead — noise then pointed one way, towards bona
# fide. Here the same channel distribution is drawn for every clip whatever
# its label, so the recording channel carries no information about the label
# at all, and the model has to find what separates the classes elsewhere.
#
# Finding 14 degraded inputs at scoring time only and made clean-fake misses
# worse, because the fixed weights had never seen a degraded fake. This is the
# training-time version of that idea, which is the one the ASVspoof DF track
# and every codec-robust system since uses.
#
# Three stages, each skipped independently, in the order a real recording
# passes through them:
#
#   reverb   p=0.3  convolve with a room impulse response (OpenSLR 28: the
#                   simulated small/medium/large rooms plus the real RIRs)
#   noise    p=0.5  MUSAN noise or music at 5-25 dB SNR, a random segment
#   codec    p=0.5  MP3 or Opus at a random quality, or 8 kHz phone band
#
# About 17.5% of clips (0.7 x 0.5 x 0.5) come through untouched, so clean audio
# is still in the training distribution. RawBoost (--rawboost), when also
# given, runs first: it models the microphone, this models what comes after.
#
# THIS IS TRAINING-ONLY, for the same reason as rawboost.py: never imported by
# model.py, app.py, evaluate.py or calibrate.py. It works on the clip already
# reduced to what the model reads (model.standard_waveform: mono, 16 kHz,
# first 4 s), which both saves the work of distorting audio the model never
# sees and keeps preprocess_waveform's own resample and truncate as no-ops.
#
# Randomness comes from numpy's global generator, as in rawboost.py, which
# PyTorch reseeds in every DataLoader worker.
#
#   python channel_aug.py check <root>   # index the corpora, time 200 clips

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from scipy import signal

from model import MAX_LEN, SAMPLE_RATE, standard_waveform

P_REVERB, P_NOISE, P_CODEC = 0.3, 0.5, 0.5
SNR_DB = (5.0, 25.0)
CODECS = ("mp3", "opus", "tel8k")
# libsndfile's compression_level: 0 is the highest bitrate, 1 the lowest.
# MP3's 1.0 is about 32 kbit/s mono; Opus's is about 6 kbit/s, harsher than
# any voice call, so both are capped below that.
CODEC_LEVEL = {"mp3": (0.0, 0.9), "opus": (0.0, 0.8)}
MUSAN_SUBSETS = ("noise", "music")
RIR_MAX = SAMPLE_RATE      # longest RIR used, 1 s; the tail past it is inaudible


def index(root):
    """The noise files (with frame counts) and RIR files under `root`."""
    root = Path(root)
    noises = []
    for subset in MUSAN_SUBSETS:
        for p in sorted((root / "musan" / subset).rglob("*.wav")):
            info = sf.info(str(p))
            if info.samplerate == SAMPLE_RATE and info.frames >= SAMPLE_RATE:
                noises.append((str(p), info.frames))
    rir_root = root / "RIRS_NOISES"
    rirs = sorted(str(p) for p in (rir_root / "simulated_rirs").rglob("*.wav"))
    # real_rirs_isotropic_noises holds noise recordings beside the RIRs;
    # only files named as impulse responses are RIRs.
    rirs += sorted(str(p) for p in (rir_root / "real_rirs_isotropic_noises").glob("*.wav")
                   if "rir" in p.name.lower())
    if not noises or not rirs:
        raise SystemExit(f"ERROR: no MUSAN noise or no RIRs under {root}. "
                         f"Run: sbatch hpc/get_channel_aug.slurm")
    return noises, rirs


def _mono16k(path, start=0, frames=-1):
    x, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = T.Resample(sr, SAMPLE_RATE)(torch.from_numpy(x)).numpy()
    return x


def reverb(x, rir):
    rir = rir[:RIR_MAX]
    # Start at the direct path, so the reverberant clip stays time-aligned.
    rir = rir[int(np.argmax(np.abs(rir))):]
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-9)
    y = signal.fftconvolve(x, rir)[:len(x)]
    return y * (np.abs(x).max() / (np.abs(y).max() + 1e-9))


def add_noise(x, noise, snr_db):
    px = np.mean(x ** 2)
    pn = np.mean(noise ** 2)
    if px == 0 or pn == 0:
        return x
    return x + noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


def codec(x, kind, level=None):
    n = len(x)
    if kind == "tel8k":
        t = torch.from_numpy(x.astype(np.float32))
        y = T.Resample(8000, SAMPLE_RATE)(T.Resample(SAMPLE_RATE, 8000)(t)).numpy()
    else:
        fmt, sub = {"mp3": ("MP3", "MPEG_LAYER_III"), "opus": ("OGG", "OPUS")}[kind]
        buf = io.BytesIO()
        sf.write(buf, np.clip(x, -1, 1).astype(np.float32), SAMPLE_RATE,
                 format=fmt, subtype=sub, compression_level=level)
        buf.seek(0)
        y, _ = sf.read(buf, dtype="float32")
    # MP3's encoder delay shifts the audio slightly; the model is trained on
    # 4 s windows with no alignment, so trim or pad back to the same length.
    return np.pad(y, (0, max(0, n - len(y))))[:n]


class ChannelAug:
    """Picklable callable for AVSpoofDataset: (channels, frames) tensor in,
    (1, <= MAX_LEN) tensor at SAMPLE_RATE out. `before` (e.g. RawBoost) runs
    first, on the clip at its own sample rate, as it always has."""

    def __init__(self, root, before=None, max_len=MAX_LEN):
        self.root = str(root)
        self.before = before
        self.max_len = max_len
        self.noises, self.rirs = index(root)

    def __call__(self, waveform, sample_rate):
        if self.before is not None:
            waveform = self.before(waveform, sample_rate)
        x = standard_waveform(waveform, sample_rate, self.max_len)[0].numpy().astype(np.float64)
        n = len(x)
        if n == 0:
            return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        rng = np.random
        if rng.rand() < P_REVERB:
            x = reverb(x, _mono16k(self.rirs[rng.randint(len(self.rirs))]))
        if rng.rand() < P_NOISE:
            path, frames = self.noises[rng.randint(len(self.noises))]
            start = rng.randint(max(1, frames - n))
            noise = _mono16k(path, start, min(n, frames))
            noise = np.resize(noise, n)          # loop a short file
            x = add_noise(x, noise, rng.uniform(*SNR_DB))
        if rng.rand() < P_CODEC:
            kind = CODECS[rng.randint(len(CODECS))]
            x = codec(x, kind, rng.uniform(*CODEC_LEVEL[kind]) if kind in CODEC_LEVEL else None)
        peak = np.abs(x).max()
        if peak > 1:
            x = x / peak
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def __repr__(self):
        return (f"ChannelAug(reverb p={P_REVERB}, noise p={P_NOISE} at {SNR_DB} dB, "
                f"codec p={P_CODEC} {CODECS}; {len(self.noises)} noise files, "
                f"{len(self.rirs)} RIRs; before={self.before})")


def _check(root):
    import time
    aug = ChannelAug(root)
    print(aug)
    per = {s: sum(1 for p, _ in aug.noises if f"/{s}/" in p) for s in MUSAN_SUBSETS}
    print(f"noise files by subset: {per}")
    np.random.seed(0)
    clean = torch.from_numpy(
        (0.1 * np.sin(np.arange(MAX_LEN) * 2 * np.pi * 220 / SAMPLE_RATE)).astype(np.float32)
    ).unsqueeze(0)
    t0 = time.perf_counter()
    changed = 0
    for _ in range(200):
        y = aug(clean, SAMPLE_RATE)
        assert y.shape == (1, MAX_LEN) and torch.isfinite(y).all()
        changed += int(not torch.allclose(y, clean))
    dt = (time.perf_counter() - t0) / 200
    print(f"200 clips: {dt * 1000:.1f} ms/clip, {changed} changed "
          f"(expected ~{200 * (1 - (1 - P_REVERB) * (1 - P_NOISE) * (1 - P_CODEC)):.0f})")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        _check(sys.argv[2])
    else:
        raise SystemExit("usage: python channel_aug.py check <root>")
