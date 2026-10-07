"""MOSS-TTS v1.5 (OpenMOSS, Apache-2.0): an 8B delay-pattern LLM over the
MOSS-Audio-Tokenizer, May 2026. Voice cloning from the prompt.

The prompt is read with soundfile and encoded here, then passed as audio
codes: the processor's own path loader calls torchaudio.load, which in this
torch release needs system FFmpeg (absent on M3)."""

from ._util import load_mono, to_numpy

REPO_CODE = "OpenMOSS-Team/MOSS-TTS-v1.5"


def load(model_id, repo):
    import torch
    from transformers import AutoModel, AutoProcessor
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation="sdpa",
                                      dtype=torch.bfloat16).to("cuda").eval()
    return dict(processor=processor, model=model)


def synthesize(m, job):
    import torch
    p = m["processor"]
    sr = int(p.model_config.sampling_rate)
    wav, _ = load_mono(job["prompt_path"], sr)
    codes = p.encode_audios_from_wav([torch.from_numpy(wav).unsqueeze(0)], sr)[0]
    conv = [[p.build_user_message(text=job["text"], reference=[codes])]]
    batch = p(conv, mode="generation")
    with torch.no_grad():
        out = m["model"].generate(input_ids=batch["input_ids"].to("cuda"),
                                  attention_mask=batch["attention_mask"].to("cuda"),
                                  max_new_tokens=4096)
    audio = next(iter(p.decode(out))).audio_codes_list[0]
    return to_numpy(audio), sr, job["speaker"]
