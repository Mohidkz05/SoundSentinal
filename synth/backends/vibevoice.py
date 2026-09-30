"""VibeVoice-Realtime 0.5B (Microsoft, MIT), the streaming model Microsoft
still supports. Preset voices: the English .pt voice prompts shipped in the
repository's demo/voices/streaming_model (MIT)."""

import copy

VOICE_URL = ("https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/"
             "streaming_model/{}.pt")
VOICES = ["en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man",
          "en-Grace_woman", "en-Mike_man", "in-Samuel_man"]


def load(model_id, repo):
    import os
    import urllib.request
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    cache = os.path.join(os.environ.get("HF_HOME", "."), "vibevoice-voices")
    os.makedirs(cache, exist_ok=True)
    prompts = {}
    for v in VOICES:
        path = os.path.join(cache, f"{v}.pt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(VOICE_URL.format(v), path)
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            prompts[v] = torch.load(path, map_location="cuda", weights_only=True)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    return dict(model=model, proc=VibeVoiceStreamingProcessor.from_pretrained(repo),
                prompts=prompts)


def synthesize(m, job):
    voice = VOICES[job["index"] % len(VOICES)]
    prompt = m["prompts"][voice]
    inputs = m["proc"].process_input_with_cached_prompt(
        text=job["text"], cached_prompt=prompt, padding=True, return_tensors="pt",
        return_attention_mask=True)
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = m["model"].generate(**inputs, max_new_tokens=None, cfg_scale=1.5,
                              tokenizer=m["proc"].tokenizer,
                              generation_config={"do_sample": False}, verbose=False,
                              all_prefilled_outputs=copy.deepcopy(prompt))
    audio = out.speech_outputs[0]
    return audio.float().cpu().numpy().reshape(-1), 24000, voice
