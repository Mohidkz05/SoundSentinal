"""Soprano 80M (ekwek, Apache-2.0). One built-in voice."""

from ._util import to_numpy


def load(model_id, repo):
    from soprano import SopranoTTS
    return SopranoTTS(backend="transformers", device="cuda")


def synthesize(model, job):
    out = model.infer(job["text"])
    sr = getattr(model, "sample_rate", 32000)
    return to_numpy(out), sr, "default"
