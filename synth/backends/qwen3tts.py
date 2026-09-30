"""Qwen3-TTS 12Hz Base, 0.6B and 1.7B (Alibaba Qwen, Apache-2.0). Voice
cloning from the prompt and its transcript."""

from ._util import to_numpy


def load(model_id, repo):
    import torch
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(repo, device_map="cuda:0", dtype=torch.bfloat16,
                                         attn_implementation="sdpa")


def synthesize(model, job):
    wavs, sr = model.generate_voice_clone(text=job["text"], language="English",
                                          ref_audio=job["prompt_path"],
                                          ref_text=job["prompt_text"])
    return to_numpy(wavs[0]), sr, job["speaker"]
