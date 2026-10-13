# Adapter: the Speech Anti-Spoofing Arena's test sets. EVALUATION ONLY.
#
# The Arena (huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/
# SpeechAntiSpoofingArena) repackages published anti-spoofing test sets as
# Parquet, one Hugging Face dataset each, and ranks systems by EER on them. Its
# In-the-Wild set is the same 31,779 clips in_the_wild.py reads, so a number
# from here sits beside other people's on the same trials.
#
# Never train, select or calibrate on anything read through this file: the
# point of a benchmark number is that the model met the clips once. calibrate.py
# has no route to it, and it is not an --extra-train source.
#
# Only sets whose licence allows commercial use are listed in SETS (checked 13
# October 2026 against each dataset card). Left out on purpose: ADD 2022 and
# ADD 2023 (CC BY-NC-ND), PyAra and XMAD (CC BY-NC-SA), SONAR (CC BY-NC) and
# J-SPAW (non-commercial). The revisions are the ones the Arena's manifest pins
# (arena-manifest, ranking v4), so the trial lists match the leaderboard's.
#
# Layout under $ARENA_ROOT/<set>/ after hpc/get_arena.slurm:
#     audio/<path>       one file per trial, bytes exactly as shipped
#     trials.csv         file, utterance_id, label, speaker_id, attack_id
#
#     python arena.py extract <set> <dir with the .parquet shards> <out dir>
#     python arena.py scores <evaluate.py JSON> <scores.txt>   # Arena format

import json
import os
import sys
from pathlib import Path

import pandas as pd

TRIALS_CSV = "trials.csv"

# name -> (pinned revision, trials, licence)
SETS = {
    "ASVspoof2021_LA": ("dc119733697c946fcd17fe7c1541d7f26b4bbe07", 181566, "ODC-By"),
    "ASVspoof2021_DF": ("16d4f7d6c68694ac9b0bd43b83df322d1bc5102e", 611829, "ODbL"),
    "ASVspoof5": ("9937de5e630b137bb31e9f5901209b2705ddef63", 680774, "ODC-By"),
    "CD-ADD": ("c2de87d49b268b624e6af7440dc2890703098965", 20786, "CC BY 4.0"),
    "CFAD": ("53d7855c1c378524f7b7b1030bcb6b2caa327fe6", 62999, "CC BY 4.0"),
    "CVoiceFake_small": ("eb60cf4d13993b583e9515db3d14c040d2404cf7", 138136, "CC BY 4.0"),
    "DeepVoice": ("cc3bdf544cfc09bd9cc788f7f022ba1af9daf701", 5053, "MIT"),
    "LibriSeVoc": ("13d69268a57afc7ebba2421d2a31730a390f1ff2", 18487, "CC BY-SA 4.0"),
    "ArAD": ("350184966eeb5b46ff2acdabd8f4d12e41e582da", 3570, "ODC-By"),
    "DECRO": ("e54c29df5b81bac75e5cc294d1ab0d4db72a5c71", 37314, "CC BY 4.0"),
    "ODSS": ("1968e6d0ef141c4572073695bdc1d17a8706177f", 26954, "CC BY-SA 4.0"),
    "HABLA": ("efb50f35a2c63928e06aacb4d97d64ccb193f7c0", 80816, "CC BY 4.0"),
    "DFADD": ("c578c836da3b522b27d3dd85f89309f1737e5d31", 3755, "MIT"),
    "EmoFake_test": ("588c7638837512cd50747c700f955ea43eb84c39", 17500, "CC BY 4.0"),
}

LABELS = ("bonafide", "spoof")   # the Arena's ClassLabel order, and LABEL_MAP's


def get_arena_root(name):
    if name not in SETS:
        raise ValueError(f"Unknown Arena set {name!r}; choose from {sorted(SETS)}.")
    env = os.getenv("ARENA_ROOT")
    base = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "arena"
    root = base / name
    if not (root / TRIALS_CSV).is_file():
        raise FileNotFoundError(
            f"No {TRIALS_CSV} under {root}.\n"
            f"Fetch and extract it first:  sbatch hpc/get_arena.slurm {name}\n"
            f"$ARENA_ROOT is currently {env or '(unset)'}.")
    return root


def load_protocol(name, root=None):
    """(protocol frame shaped like an ASVspoof one, audio root) for one set."""
    root = Path(root) if root is not None else get_arena_root(name)
    trials = pd.read_csv(root / TRIALS_CSV, dtype=str, keep_default_na=False)
    expected = SETS[name][1]
    if len(trials) != expected:
        raise ValueError(
            f"{name}: {len(trials)} trials in {root / TRIALS_CSV}, the Arena lists "
            f"{expected}. A partial extract would give an EER that compares to nothing.")
    frame = pd.DataFrame({
        "speaker_id": trials["speaker_id"],
        "audio_file_name": trials["file"],
        "_": "-",
        "system_id": trials["attack_id"].where(trials["label"] == "spoof", "-"),
        "label": trials["label"],
    })
    frame.attrs["utterance_id"] = trials["utterance_id"].tolist()
    return frame, root


def extract(name, shard_dir, out_dir):
    """Unpack the audio embedded in the Parquet shards into files, unchanged."""
    import pyarrow.parquet as pq

    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    rows = []
    for shard in sorted(shard_dir.glob("test-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=256, columns=["path", "audio", "label", "notes"]):
            for rec in batch.to_pylist():
                notes = json.loads(rec["notes"]) if rec["notes"] else {}
                rel = Path("audio") / rec["path"]
                target = out_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                data = rec["audio"]["bytes"]
                if not (target.is_file() and target.stat().st_size == len(data)):
                    target.write_bytes(data)
                label = rec["label"]
                rows.append({
                    "file": str(rel),
                    "utterance_id": notes.get("utterance_id") or Path(rec["path"]).stem,
                    "label": LABELS[label] if isinstance(label, int) else str(label),
                    "speaker_id": notes.get("speaker_id") or "-",
                    "attack_id": (notes.get("attack_id") or notes.get("attack")
                                  or notes.get("vocoder") or "-"),
                })
        print(f"  {shard.name}: done, {len(rows)} trials so far", flush=True)
    frame = pd.DataFrame(rows)
    expected = SETS[name][1]
    if len(frame) != expected or frame["file"].duplicated().any():
        raise SystemExit(f"ERROR: {name}: extracted {len(frame)} trials "
                         f"({frame['file'].nunique()} distinct files), expected {expected}.")
    frame.to_csv(out_dir / TRIALS_CSV, index=False)
    print(f"Wrote {len(frame)} trials to {out_dir / TRIALS_CSV}: "
          f"{frame['label'].value_counts().to_dict()}")


def write_scores(result_json, out_file):
    """evaluate.py's JSON -> the Arena's scores.txt: `<utterance_id> <score>`,
    higher = more bonafide. Ours is log-odds of spoof, so the sign flips."""
    result = json.loads(Path(result_json).read_text())
    ids, scores = result["utterance_ids"], result["scores"]
    if len(ids) != len(scores):
        raise SystemExit(f"ERROR: {len(ids)} utterance ids for {len(scores)} scores.")
    Path(out_file).write_text("".join(f"{u} {-s:.6f}\n" for u, s in zip(ids, scores)))
    print(f"Wrote {len(ids)} scores to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4 and sys.argv[1] == "scores":
        write_scores(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: python arena.py extract <set> <shard dir> <out dir>\n"
                         "       python arena.py scores <evaluate.py JSON> <scores.txt>")
