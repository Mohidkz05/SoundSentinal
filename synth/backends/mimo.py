"""MiMo-Audio 7B Instruct (Xiaomi, MIT) with its own MiMo-Audio-Tokenizer,
September 2025. Voice cloning via tts_sft(prompt_speech=).

The code is not a package: the repository is cloned once beside the
environment and imported from there."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

GIT = "https://github.com/XiaomiMiMo/MiMo-Audio.git"
TOKENIZER = "XiaomiMiMo/MiMo-Audio-Tokenizer"


def load(model_id, repo):
    from huggingface_hub import snapshot_download
    code = Path(os.environ.get("SYNTH_ENV_DIR", ".")) / "mimo-src"
    if not code.is_dir():
        subprocess.run(["git", "clone", "--depth", "1", GIT, str(code)], check=True)
    sys.path.insert(0, str(code))
    from src.mimo_audio.mimo_audio import MimoAudio
    return MimoAudio(snapshot_download(repo), snapshot_download(TOKENIZER))


def synthesize(model, job):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.wav")
        model.tts_sft(job["text"], out, prompt_speech=job["prompt_path"])
        audio, sr = sf.read(out, dtype="float32")
    return audio, sr, job["speaker"]
