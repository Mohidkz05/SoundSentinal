"""Marvis TTS 250M v0.1 (Marvis AI, Apache-2.0), a Sesame-CSM-architecture
model, through transformers. Voice cloning: the prompt is given as a context
turn by speaker 0 and the model continues as that speaker."""

from ._util import load_mono, to_numpy


def load(model_id, repo):
    from transformers import AutoProcessor, CsmForConditionalGeneration
    return dict(processor=AutoProcessor.from_pretrained(repo),
                model=CsmForConditionalGeneration.from_pretrained(repo, device_map="cuda"))


def synthesize(m, job):
    import torch
    prompt, _ = load_mono(job["prompt_path"], 24000)
    conversation = [
        {"role": "0", "content": [{"type": "text", "text": job["prompt_text"]},
                                  {"type": "audio", "path": prompt}]},
        {"role": "0", "content": [{"type": "text", "text": job["text"]}]},
    ]
    inputs = m["processor"].apply_chat_template(conversation, tokenize=True,
                                                return_dict=True).to("cuda")
    with torch.no_grad():
        # The checkpoint's default is 125 frames (10 s at 12.5 Hz), which cuts
        # long sentences off; allow ~1 s per 10 characters plus 3 s, up to 30 s.
        frames = int(12.5 * min(30, 3 + len(job["text"]) / 10))
        audio = m["model"].generate(**inputs, output_audio=True, max_new_tokens=frames)
    return to_numpy(audio[0]), 24000, job["speaker"]
