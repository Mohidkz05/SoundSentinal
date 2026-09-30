"""Chatterbox and Chatterbox-Turbo (Resemble AI, MIT). Voice cloning from the
LibriSpeech prompt. Both embed Resemble's Perth watermark, as every
Chatterbox clip in the wild does."""

from ._util import to_numpy


def load(model_id, repo):
    if model_id == "chatterbox-turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
    else:
        from chatterbox.tts import ChatterboxTTS as Model
    return Model.from_pretrained(device="cuda")


def synthesize(model, job):
    wav = model.generate(job["text"], audio_prompt_path=job["prompt_path"])
    return to_numpy(wav), model.sr, job["speaker"]
