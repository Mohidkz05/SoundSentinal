"""Piper (rhasspy, MIT) with three voices whose training data allows
commercial use: LJSpeech (public domain), LibriTTS-R (CC BY 4.0; multi-speaker,
a speaker per clip in turn) and Cori (LibriVox, public domain). Excluded:
voices trained on non-commercial data (e.g. hfc_female, lessac)."""

VOICES = {"piper-ljspeech": "en/en_US/ljspeech/medium/en_US-ljspeech-medium",
          "piper-libritts-r": "en/en_US/libritts_r/medium/en_US-libritts_r-medium",
          "piper-cori": "en/en_GB/cori/high/en_GB-cori-high"}


def load(model_id, repo):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    stem = VOICES[model_id]
    onnx = hf_hub_download(repo, f"{stem}.onnx")
    hf_hub_download(repo, f"{stem}.onnx.json")
    return PiperVoice.load(onnx)


def synthesize(voice, job):
    import numpy as np
    from piper import SynthesisConfig
    n = voice.config.num_speakers
    spk = job["index"] % n if n > 1 else None
    chunks = voice.synthesize(job["text"], syn_config=SynthesisConfig(speaker_id=spk))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    return audio, voice.config.sample_rate, f"speaker-{spk}" if spk is not None else "default"
