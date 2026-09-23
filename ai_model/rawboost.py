# rawboost.py — training-time data augmentation on the raw waveform.
#
# RawBoost (Tak, Kamble, Patino, Todisco, Evans; ICASSP 2022), ported from the
# official implementation in github.com/TakHemlata/SSL_Anti-spoofing (MIT
# licence, © 2022 Hemlata Tak), which is also where the default parameters
# below come from.
#
# Why a port rather than a copy: upstream's randRange calls int() on a
# one-element array, which numpy 2.x refuses, so it does not run on our pinned
# numpy 2.5. Verified 23 September 2026: with that one line patched, upstream
# and this file produce bit-identical output (max difference 0.0) for all eight
# algos across three seeds, consuming the random stream in the same order.
# Throughput is not a concern — about 16 ms per clip for algo 5 on the laptop,
# so 8 DataLoader workers outrun AASIST's ~60 clips/s by nearly tenfold. It distorts each training clip
# a little differently every time it is loaded, so the model cannot rely on the
# studio-clean channel ASVspoof2019 was recorded through. That is the failure
# RESULTS.md Finding 6 measured: 3.17% EER on LA eval, 37.15% on In-the-Wild,
# where clips have been compressed, re-recorded and uploaded.
#
# Three kinds of distortion, each modelling something a real channel does:
#
#   1  Linear and non-linear convolutive noise — random band-stop filters plus
#      harmonic distortion. What a cheap microphone, a codec or a loudspeaker
#      does to the spectrum.
#   2  Impulsive signal-dependent noise — a random handful of samples nudged in
#      proportion to their own value. Clipping and quantisation artefacts.
#   3  Stationary signal-independent noise — filtered background noise at a
#      random SNR. A room, a fan, a line hum.
#
# `algo` picks which, following the upstream numbering so a result can be
# compared against theirs:
#
#   1, 2, 3    one of the above          4  1 then 2 then 3
#   5          1 then 2                  6  1 then 3
#   7          2 then 3                  8  1 and 2 in parallel, summed
#
# Upstream recommends 5 for LA-style data and 3 for the DF track, whose clips
# pass through lossy codecs — the closer of the two to In-the-Wild.
#
# THIS IS TRAINING-ONLY. It is deliberately not in model.py and never reaches
# app.py or evaluate.py: augmentation exists to make training harder, and
# applying it at inference would make every served reading noisier for no
# reason. Train/serve parity (verify_setup.py) is about the deterministic
# preprocessing that follows this step, and that is untouched.
#
# Randomness comes from numpy's global generator, as upstream. PyTorch seeds it
# separately in every DataLoader worker and every epoch, so workers do not
# produce identical distortions — verify_setup.py checks that this holds.

import copy

import numpy as np
import torch
from scipy import signal

ALGOS = {
    1: "convolutive",
    2: "impulsive",
    3: "stationary noise",
    4: "convolutive + impulsive + stationary noise",
    5: "convolutive + impulsive",
    6: "convolutive + stationary noise",
    7: "impulsive + stationary noise",
    8: "convolutive + impulsive, in parallel",
}

# Upstream defaults (SSL_Anti-spoofing main.py), unchanged.
DEFAULTS = dict(
    # Convolutive noise and the filter behind stationary noise
    N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
    minCoeff=10, maxCoeff=100, minG=0, maxG=0,
    minBiasLinNonLin=5, maxBiasLinNonLin=20,
    # Impulsive noise: up to P% of samples, scaled by g_sd
    P=10, g_sd=2,
    # Stationary noise SNR range, dB
    SNRmin=10, SNRmax=40,
)


def _rand_range(lo, hi, integer):
    y = np.random.uniform(low=lo, high=hi)
    return int(y) if integer else y


def _norm_wav(x, always):
    peak = np.amax(np.abs(x))
    if peak == 0:
        return x
    if always or peak > 1:
        x = x / peak
    return x


def _notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    """A random cascade of FIR band-pass filters with a random overall gain."""
    b = 1
    for _ in range(nBands):
        fc = _rand_range(minF, maxF, False)
        bw = _rand_range(minBW, maxBW, False)
        c = _rand_range(minCoeff, maxCoeff, True)
        if c % 2 == 0:  # firwin needs an odd tap count for a band-pass
            c += 1
        f1 = max(fc - bw / 2, 1e-3)
        f2 = min(fc + bw / 2, fs / 2 - 1e-3)
        b = np.convolve(signal.firwin(c, [f1, f2], window="hamming", fs=fs), b)
    gain = _rand_range(minG, maxG, False)
    _, h = signal.freqz(b, 1, fs=fs)
    return pow(10, gain / 20) * b / np.amax(np.abs(h))


def _filter_fir(x, b):
    """Zero-phase-delay FIR filtering: pad, filter, trim the group delay."""
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n), "constant"))
    return y[int(n / 2):int(y.shape[0] - n / 2)]


def convolutive(x, fs, N_f, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, **_):
    y = np.zeros(x.shape[0])
    for i in range(N_f):
        if i == 1:
            minG -= minBiasLinNonLin
            maxG -= maxBiasLinNonLin
        b = _notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
        y = y + _filter_fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm_wav(y, False)


def impulsive(x, P, g_sd, **_):
    beta = _rand_range(0, P, False)
    y = copy.deepcopy(x)
    n = int(x.shape[0] * (beta / 100))
    p = np.random.permutation(x.shape[0])[:n]
    f_r = (2 * np.random.rand(n) - 1) * (2 * np.random.rand(n) - 1)
    y[p] = x[p] + g_sd * x[p] * f_r
    return _norm_wav(y, False)


def stationary_noise(x, fs, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW,
                     minCoeff, maxCoeff, minG, maxG, **_):
    noise = np.random.normal(0, 1, x.shape[0])
    b = _notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = _norm_wav(_filter_fir(noise, b), True)
    snr = _rand_range(SNRmin, SNRmax, False)
    x_norm = np.linalg.norm(x, 2)
    if x_norm == 0:
        return x
    noise = noise / np.linalg.norm(noise, 2) * x_norm / 10.0 ** (0.05 * snr)
    return x + noise


def rawboost(x, fs, algo, **overrides):
    """Distort one mono waveform (1-D numpy array). Returns the same length."""
    p = {**DEFAULTS, **overrides}
    if algo == 1:
        return convolutive(x, fs, **p)
    if algo == 2:
        return impulsive(x, **p)
    if algo == 3:
        return stationary_noise(x, fs, **p)
    if algo == 4:
        return stationary_noise(impulsive(convolutive(x, fs, **p), **p), fs, **p)
    if algo == 5:
        return impulsive(convolutive(x, fs, **p), **p)
    if algo == 6:
        return stationary_noise(convolutive(x, fs, **p), fs, **p)
    if algo == 7:
        return stationary_noise(impulsive(x, **p), fs, **p)
    if algo == 8:
        return _norm_wav(convolutive(x, fs, **p) + impulsive(x, **p), False)
    raise ValueError(f"RawBoost algo must be one of {sorted(ALGOS)}, got {algo!r}")


class RawBoost:
    """Picklable callable for AVSpoofDataset: (channels, frames) tensor in, same out.

    A class rather than a closure because DataLoader workers pickle the dataset.
    """

    def __init__(self, algo):
        if algo not in ALGOS:
            raise ValueError(f"RawBoost algo must be one of {sorted(ALGOS)}, got {algo!r}")
        self.algo = algo

    def __call__(self, waveform, sample_rate):
        mono = waveform.mean(dim=0).numpy().astype(np.float64)
        out = rawboost(mono, sample_rate, self.algo)
        return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)

    def __repr__(self):
        return f"RawBoost(algo={self.algo}: {ALGOS[self.algo]})"
