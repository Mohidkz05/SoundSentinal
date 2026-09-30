"""Kokoro-82M (hexgrad, Apache-2.0). Preset voices: American (a) and British (b)
English, cycled by clip index."""

VOICES = ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
          "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river", "am_michael",
          "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx",
          "am_adam", "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george",
          "bm_fable", "bm_lewis", "bm_daniel"]


def load(model_id, repo):
    from kokoro import KPipeline
    return {lang: KPipeline(lang_code=lang, repo_id=repo) for lang in "ab"}


def synthesize(pipes, job):
    import numpy as np
    voice = VOICES[job["index"] % len(VOICES)]
    chunks = [a for _, _, a in pipes[voice[0]](job["text"], voice=voice)]
    return np.concatenate([np.asarray(c) for c in chunks]), 24000, voice
