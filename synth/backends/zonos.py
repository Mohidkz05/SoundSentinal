"""Zonos v0.1 transformer (Zyphra, Apache-2.0). Voice cloning from a speaker
embedding of the prompt."""

from ._util import load_mono, to_numpy


def load(model_id, repo):
    from ._util import use_bundled_espeak
    use_bundled_espeak()
    from zonos.model import Zonos
    return dict(model=Zonos.from_pretrained(repo, device="cuda"), speakers={})


def synthesize(m, job):
    import torch
    from zonos.conditioning import make_cond_dict
    model = m["model"]
    # One embedding per prompt file, computed once.
    if job["prompt_path"] not in m["speakers"]:
        wav, sr = load_mono(job["prompt_path"])
        m["speakers"][job["prompt_path"]] = model.make_speaker_embedding(
            torch.from_numpy(wav).unsqueeze(0), sr)
    cond = make_cond_dict(text=job["text"], speaker=m["speakers"][job["prompt_path"]],
                          language="en-us")
    with torch.no_grad():
        codes = model.generate(model.prepare_conditioning(cond))
        wav = model.autoencoder.decode(codes).cpu()[0]
    return to_numpy(wav), model.autoencoder.sampling_rate, job["speaker"]
