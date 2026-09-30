"""OuteTTS 1.0 0.6B (OuteAI, Apache-2.0), Hugging Face backend. Voice cloning:
a speaker profile per prompt file, built once. create_speaker transcribes the
prompt with Whisper (MIT); the profile is cached so that runs once per file."""

from ._util import to_numpy


def load(model_id, repo):
    import outetts
    iface = outetts.Interface(outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_0_6B, backend=outetts.Backend.HF))
    return dict(iface=iface, speakers={})


def synthesize(m, job):
    import outetts
    iface = m["iface"]
    if job["prompt_path"] not in m["speakers"]:
        m["speakers"][job["prompt_path"]] = iface.create_speaker(job["prompt_path"])
    out = iface.generate(outetts.GenerationConfig(
        text=job["text"], speaker=m["speakers"][job["prompt_path"]],
        generation_type=outetts.GenerationType.CHUNKED))
    return to_numpy(out.audio), out.sr, job["speaker"]
