"""Generate one model's fakes for one split, resumably.

    python generate.py --family kokoro --model kokoro-82m --split train --n 6000

Runs inside that family's own environment (hpc/synth.slurm builds it from
envs/<family>.txt), because the sixteen generators pin incompatible versions of
torch and transformers. Writes $SYNTH_ROOT/<split>/<family>/<model>/NNNNNN.flac
and a manifest.csv beside them; clips already on disk are skipped, so a job cut
off at its time limit continues where it stopped.

The registry and the split live in ai_model/synth.py, which this reads without
importing the rest of ai_model (whose torch may not be this environment's).
"""

import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ai_model"))
import jobs as joblist  # noqa: E402
import synth  # noqa: E402  (ai_model/synth.py: numpy + pandas only)

MIN_S, MAX_S = 1.0, 30.0          # a clip outside this is a generation failure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(synth.FAMILIES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, choices=synth.SPLITS)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SYNTH_ROOT", "synth-out")))
    ap.add_argument("--librispeech", type=Path, default=Path(os.environ["LIBRISPEECH_ROOT"]))
    args = ap.parse_args()

    entry = {m[0]: m for m in synth.FAMILIES[args.family]}.get(args.model)
    if entry is None:
        sys.exit(f"{args.model} is not a {args.family} model: {list(synth.FAMILIES[args.family])}")
    if args.split in synth.PROBES:
        if args.family not in synth.splits()[args.split]:
            sys.exit(f"{args.split} probes only {synth.splits()[args.split]}")
    elif synth.split_of(args.family) != args.split:
        sys.exit(f"{args.family} belongs to {synth.split_of(args.family)}, not {args.split}")
    _, repo, licence, voice = entry

    speakers = None
    if args.split != "train":
        import librispeech  # ai_model/librispeech.py
        clips, _ = librispeech.load_clips(args.librispeech)
        speakers = synth.heldout_speakers(args.split, clips)
    todo = joblist.build(args.split, args.model, args.n, args.librispeech, speakers)

    out = args.out / args.split / args.family / args.model
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    done = set()
    if manifest.is_file():
        with manifest.open() as f:
            done = {row["file"] for row in csv.DictReader(f)}
    todo = [j for j in todo if f"{args.split}/{args.family}/{args.model}/{j['index']:06d}.flac"
            not in done]
    print(f"{args.model} ({repo}, {licence}, {voice} voices): {len(done)} done, "
          f"{len(todo)} to go", flush=True)
    if not todo:
        return

    backend = importlib.import_module(f"backends.{args.family}")
    model = backend.load(args.model, repo)
    new = not manifest.is_file()
    failures, t0 = 0, time.monotonic()
    with manifest.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "voice", "text", "text_utt",
                                          "prompt", "seconds", "sample_rate"])
        if new:
            w.writeheader()
        for k, job in enumerate(todo, 1):
            try:
                audio, sr, voice_name = backend.synthesize(model, job)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                secs = len(audio) / sr
                if not (MIN_S <= secs <= MAX_S) or not np.isfinite(audio).all() \
                        or np.abs(audio).max() < 1e-3:
                    raise ValueError(f"bad output: {secs:.1f} s, peak {np.abs(audio).max():.4f}")
            except Exception as exc:  # noqa: BLE001 — one bad clip must not end the run
                failures += 1
                print(f"  clip {job['index']}: {type(exc).__name__}: {exc}", flush=True)
                if failures == 1:
                    import traceback
                    traceback.print_exc()
                if failures > max(20, len(todo) // 10):
                    sys.exit(f"ERROR: {failures} failures; stopping. Fix the backend.")
                continue
            peak = np.abs(audio).max()
            if peak > 1:
                audio = audio / peak
            rel = f"{args.split}/{args.family}/{args.model}/{job['index']:06d}.flac"
            sf.write(args.out / rel, audio, sr, subtype="PCM_16")
            w.writerow(dict(file=rel, speaker=job["speaker"], voice=voice_name,
                            text=job["text"], text_utt=job["text_utt"],
                            prompt=Path(job["prompt_path"]).name if voice == "clone" else "",
                            seconds=f"{secs:.2f}", sample_rate=sr))
            f.flush()
            if k % 100 == 0 or k == len(todo):
                rate = k / (time.monotonic() - t0)
                print(f"  {k}/{len(todo)} | {rate:.2f} clips/s | {failures} failed | "
                      f"ETA {(len(todo) - k) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
