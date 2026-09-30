"""Kitten TTS nano 0.2 (KittenML, Apache-2.0). Its eight preset voices in turn."""

VOICES = ["expr-voice-2-m", "expr-voice-2-f", "expr-voice-3-m", "expr-voice-3-f",
          "expr-voice-4-m", "expr-voice-4-f", "expr-voice-5-m", "expr-voice-5-f"]


def load(model_id, repo):
    from ._util import use_bundled_espeak
    use_bundled_espeak()
    from kittentts import KittenTTS
    return KittenTTS(repo)


def synthesize(model, job):
    voice = VOICES[job["index"] % len(VOICES)]
    return model.generate(job["text"], voice=voice), 24000, voice
