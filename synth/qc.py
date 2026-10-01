"""Intelligibility check for generated fakes (RESULTS.md Finding 18).

    python qc.py <split>/<family>/<model> [--n 50]

Transcribes up to n clips (a fixed sample) with Whisper small (OpenAI, MIT)
and scores word error rate against the text the generator was asked to say.
Writes qc.json beside the manifest. A model whose median WER exceeds
MAX_MEDIAN_WER is producing babble, not speech, and is dropped (the rule is
pre-registered); clips are never filtered one by one.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_MEDIAN_WER = 0.30


def norm(text):
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    import jiwer
    import torch
    from scipy.signal import resample_poly
    import soundfile as sf
    from transformers import pipeline

    root = Path(os.environ["SYNTH_ROOT"])
    d = root / args.model_dir
    clips = pd.read_csv(d / "manifest.csv", dtype=str)
    sample = clips.sample(n=min(args.n, len(clips)), random_state=0)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                   device=0 if torch.cuda.is_available() else -1)
    wers = []
    for _, row in sample.iterrows():
        x, sr = sf.read(root / row["file"], dtype="float32")
        if sr != 16000:
            from math import gcd
            g = gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        hyp = asr({"raw": x, "sampling_rate": 16000},
                  generate_kwargs={"language": "en", "task": "transcribe"})["text"]
        ref = " ".join(norm(row["text"]))
        wers.append(jiwer.wer(ref, " ".join(norm(hyp)) or "<empty>"))
    report = dict(model_dir=args.model_dir, n=len(wers), median_wer=float(np.median(wers)),
                  mean_wer=float(np.mean(wers)), passes=bool(np.median(wers) <= MAX_MEDIAN_WER),
                  max_median_wer=MAX_MEDIAN_WER)
    (d / "qc.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
