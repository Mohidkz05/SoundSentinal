"""Maya1 (Maya Research, Apache-2.0) with the SNAC 24 kHz codec (MIT). Preset
voices: text descriptions of speaker age, accent, pitch and pace, varied by
clip index. Prompt format and token ids are the model card's."""

import numpy as np

CODE_START, CODE_END, CODE_OFFSET = 128257, 128258, 128266
SNAC_MIN, SNAC_MAX = 128266, 156937
SOH, EOH, SOA, TEXT_EOT = 128259, 128260, 128261, 128009

GENDERS = ["Male", "Female"]
AGES = ["20s", "30s", "40s", "50s", "60s"]
ACCENTS = ["American", "British", "Australian", "Canadian", "Irish"]
TRAITS = ["normal pitch, warm timbre, conversational pacing",
          "low pitch, calm tone, slow pacing",
          "high pitch, bright timbre, energetic pacing",
          "neutral tone, clear diction, measured pacing"]


def describe(i):
    return (f"Realistic {GENDERS[i % 2].lower()} voice in the {AGES[(i // 2) % 5]} age "
            f"with {ACCENTS[(i // 10) % 5]} accent. {TRAITS[(i // 50) % 4].capitalize()}.")


def load(model_id, repo):
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return dict(model=AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                           device_map="cuda"),
                tok=AutoTokenizer.from_pretrained(repo),
                snac=SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda())


def synthesize(m, job):
    import torch
    tok = m["tok"]
    desc = describe(job["index"])
    prompt = (tok.decode([SOH]) + tok.bos_token + f'<description="{desc}"> {job["text"]}'
              + tok.decode([TEXT_EOT]) + tok.decode([EOH]) + tok.decode([SOA])
              + tok.decode([CODE_START]))
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = m["model"].generate(**inputs, max_new_tokens=2048, min_new_tokens=28,
                                  temperature=0.4, top_p=0.9, repetition_penalty=1.1,
                                  do_sample=True, eos_token_id=CODE_END,
                                  pad_token_id=tok.pad_token_id)
    ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    ids = ids[:ids.index(CODE_END)] if CODE_END in ids else ids
    codes = [t for t in ids if SNAC_MIN <= t <= SNAC_MAX]
    frames = len(codes) // 7
    l1, l2, l3 = [], [], []
    for f in range(frames):
        s = [(c - CODE_OFFSET) % 4096 for c in codes[f * 7:(f + 1) * 7]]
        l1.append(s[0]); l2 += [s[1], s[4]]; l3 += [s[2], s[3], s[5], s[6]]
    levels = [torch.tensor(x, dtype=torch.long, device="cuda").unsqueeze(0) for x in (l1, l2, l3)]
    with torch.inference_mode():
        audio = m["snac"].decoder(m["snac"].quantizer.from_codes(levels))[0, 0]
    # The card trims the codec's warm-up.
    return audio.float().cpu().numpy()[2048:], 24000, desc
