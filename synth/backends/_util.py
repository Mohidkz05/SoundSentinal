"""Shared helpers for the backends."""

import numpy as np
import soundfile as sf


def load_mono(path, sr=None):
    """A prompt file as mono float32, resampled to `sr` if given."""
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr is not None and rate != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
        rate = sr
    return x, rate


def to_numpy(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def use_bundled_espeak():
    """Point phonemizer at the espeak-ng that espeakng-loader ships in its
    wheel: M3 has no system espeak-ng and installing one needs root."""
    import os
    import espeakng_loader
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
    os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:  # noqa: BLE001 — older phonemizer has no set_data_path
        pass
