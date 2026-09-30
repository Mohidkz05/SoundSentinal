"""The deterministic list of what each generator says, and in whose voice.

Every clip is one LibriSpeech transcript read by one speaker. For voice-cloning
models the prompt is another utterance by that speaker (4-10 s, never the one
whose text is read), with its transcript for models that want it. Preset-voice
models ignore the prompt and pick a voice from their own list by clip index.

The job list depends only on (split, model id, n), so a re-run regenerates the
same clips and a resumed run skips the ones already on disk.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

MIN_CHARS, MAX_CHARS = 40, 240        # text: a sentence or two
PROMPT_S = (4.0, 10.0)                # prompt duration window, seconds
SUBSET = {"train": "train-clean-100", "heldout-a": "test-clean", "heldout-b": "test-clean"}


def utterances(librispeech_root, subset):
    """(utt_id, speaker, path, text) for every utterance in a subset."""
    base = Path(librispeech_root) / "LibriSpeech" / subset
    rows = []
    for trans in sorted(base.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            rows.append((utt, f"LS{utt.split('-')[0]}", str(trans.parent / f"{utt}.flac"),
                         text.capitalize()))
    return pd.DataFrame(rows, columns=["utt", "speaker", "path", "text"])


def build(split, model_id, n, librispeech_root, speakers=None):
    """n jobs: (index, speaker, text, text_utt, prompt_path, prompt_text)."""
    utts = utterances(librispeech_root, SUBSET[split])
    if speakers is not None:
        utts = utts[utts["speaker"].isin(speakers)]
    seed = int(hashlib.sha256(f"{split}/{model_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)

    texts = utts[utts["text"].str.len().between(MIN_CHARS, MAX_CHARS)].reset_index(drop=True)
    by_speaker = {s: g for s, g in utts.groupby("speaker")}
    durations = {}
    jobs = []
    for i in range(n):
        t = texts.iloc[rng.randint(len(texts))]
        pool = by_speaker[t["speaker"]]
        pool = pool[pool["utt"] != t["utt"]]
        # Draw prompts until one is inside the window; durations are cached.
        prompt = None
        for j in rng.permutation(len(pool))[:50]:
            p = pool.iloc[j]
            if p["path"] not in durations:
                durations[p["path"]] = sf.info(p["path"]).duration
            if PROMPT_S[0] <= durations[p["path"]] <= PROMPT_S[1]:
                prompt = p
                break
        if prompt is None:
            continue
        jobs.append(dict(index=i, speaker=t["speaker"], text=t["text"], text_utt=t["utt"],
                         prompt_path=prompt["path"], prompt_text=prompt["text"]))
    return jobs
