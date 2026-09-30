"""Dia 1.6B (Nari Labs, Apache-2.0), June 2025 checkpoint, through its
transformers port. Voice cloning: the prompt's audio and transcript are given
as the first [S1] turn and the model continues in that voice."""

from ._util import load_mono, to_numpy


def load(model_id, repo):
    from transformers import AutoProcessor, DiaForConditionalGeneration
    return dict(processor=AutoProcessor.from_pretrained(repo),
                model=DiaForConditionalGeneration.from_pretrained(repo).cuda().eval())


def synthesize(m, job):
    import torch
    proc, model = m["processor"], m["model"]
    prompt, _ = load_mono(job["prompt_path"], 44100)
    text = f"[S1] {job['prompt_text']} [S1] {job['text']}"
    inputs = proc(text=[text], audio=[prompt], padding=True, return_tensors="pt").to("cuda")
    prompt_len = proc.get_audio_prompt_len(inputs["decoder_attention_mask"])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0,
                             temperature=1.8, top_p=0.90, top_k=45)
    audio = proc.batch_decode(out, audio_prompt_len=prompt_len)[0]
    return to_numpy(audio), 44100, job["speaker"]
