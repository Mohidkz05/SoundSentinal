"""Bark (Suno, MIT): a GPT-style model over EnCodec tokens, 2023. Preset
voices: the ten English v2 speaker prompts that ship with the model."""

from ._util import to_numpy

VOICES = [f"v2/en_speaker_{i}" for i in range(10)]


def load(model_id, repo):
    import torch
    from transformers import AutoProcessor, BarkModel
    return dict(processor=AutoProcessor.from_pretrained(repo),
                model=BarkModel.from_pretrained(repo, torch_dtype=torch.float16).cuda().eval())


def synthesize(m, job):
    import torch
    voice = VOICES[job["index"] % len(VOICES)]
    inputs = m["processor"](job["text"], voice_preset=voice, return_tensors="pt")
    # The voice preset arrives as a nested dict that BatchFeature.to() leaves
    # on the CPU; move every tensor by hand.
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            moved[k] = v.to("cuda")
        elif isinstance(v, dict):
            moved[k] = {kk: vv.to("cuda") if hasattr(vv, "to") else vv for kk, vv in v.items()}
        else:
            moved[k] = v
    with torch.no_grad():
        audio = m["model"].generate(**moved, do_sample=True)
    return to_numpy(audio[0]), m["model"].generation_config.sample_rate, voice
