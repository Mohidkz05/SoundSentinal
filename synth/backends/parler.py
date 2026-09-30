"""Parler-TTS Mini v1 (Hugging Face, Apache-2.0). Preset voices: its named
speakers, described in text, with varied pace and recording quality."""

SPEAKERS = ["Jon", "Lea", "Gary", "Jenna", "Mike", "Laura", "Rick", "Eileen", "Will",
            "Karen", "Brenda", "Patrick", "Rose", "Jerry", "Yann", "Emily", "David"]
STYLES = ["speaks at a moderate pace with a clear, close-sounding recording.",
          "speaks slightly fast in a very clear recording with almost no background noise.",
          "delivers the words slowly and expressively; the recording is clean.",
          "speaks in a monotone, moderate pace; the recording is slightly distant."]


def load(model_id, repo):
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    return dict(model=ParlerTTSForConditionalGeneration.from_pretrained(repo).cuda().eval(),
                tok=AutoTokenizer.from_pretrained(repo))


def synthesize(m, job):
    import torch
    spk = SPEAKERS[job["index"] % len(SPEAKERS)]
    desc = f"{spk} {STYLES[(job['index'] // len(SPEAKERS)) % len(STYLES)]}"
    d = m["tok"](desc, return_tensors="pt").input_ids.cuda()
    p = m["tok"](job["text"], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = m["model"].generate(input_ids=d, prompt_input_ids=p)
    return out.cpu().float().numpy().reshape(-1), m["model"].config.sampling_rate, spk
