# voxpopuli.py
#
# VoxPopuli English (Wang et al., ACL 2021): speeches in the European
# Parliament, 2009-2020, recorded through the chamber's microphones and
# broadcast. CC0.
#
# WHY IT IS HERE. It is CALIBRATION data for the decision threshold, never
# training data. The LA + SpeechFake model ranks In-the-Wild well (2.65% EER)
# but real-world recordings score far higher on "spoof" than clean speech, so
# a threshold fitted on clean speech flags genuine clips (RESULTS.md Finding 9).
# Common Voice — people reading at home — got In-the-Wild's false flags from
# 46% to 15.6%. In-the-Wild's genuine clips are mostly speeches, interviews
# and broadcast audio; parliamentary speeches are the closest freely
# licensed match. None of In-the-Wild's 58 speakers sat in the European
# Parliament (checked against its meta.csv on 27 September 2026).
#
# CAVEAT. XLS-R, the model's front-end, was pretrained on unlabelled
# VoxPopuli audio (as it was on Common Voice). The model has not seen these
# clips with labels, but its features have seen this kind of audio. That
# could make these clips look more "familiar", i.e. score lower than truly
# unseen real speech — which would bias the threshold low.
#
#     python voxpopuli.py extract <dir with the .parquet shards> <out dir>
#
# writes one audio file per clip and a clips.csv (file, speaker_id, split),
# which calibrate.py reads. Fetch the shards with hpc/get_voxpopuli.slurm.

import io
import os
import sys
from pathlib import Path

import pandas as pd

CLIPS_CSV = "clips.csv"


def get_voxpopuli_root():
    env = os.getenv("VOXPOPULI_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "voxpopuli"
    if not (root / CLIPS_CSV).is_file():
        raise FileNotFoundError(
            f"No {CLIPS_CSV} under {root}.\n"
            f"Fetch and extract it first:  sbatch hpc/get_voxpopuli.slurm\n"
            f"$VOXPOPULI_ROOT is currently {env or '(unset)'}.")
    return root


def load_clips(root=None):
    """(frame with file + speaker_id, root). Paths in `file` are relative to root."""
    root = Path(root) if root is not None else get_voxpopuli_root()
    return pd.read_csv(root / CLIPS_CSV, dtype=str), root


def extract(shard_dir, out_dir):
    """Unpack the audio embedded in the Parquet shards into files.

    The `audio` column is a struct of the original file's bytes and name, so
    the bytes are written out unchanged — no decode and re-encode — and read
    later through model.load_audio like every other corpus.
    """
    import pyarrow.parquet as pq

    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    rows = []
    for shard in sorted(shard_dir.glob("*.parquet")):
        split = shard.name.split("-")[0]
        audio_dir = out_dir / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(shard, columns=["audio_id", "audio", "speaker_id", "gender"])
        for rec in table.to_pylist():
            audio = rec["audio"]
            ext = Path(audio.get("path") or "x.wav").suffix or ".wav"
            rel = Path("audio") / split / f"{rec['audio_id']}{ext}"
            (out_dir / rel).write_bytes(audio["bytes"])
            rows.append({"file": str(rel), "speaker_id": rec["speaker_id"] or "-",
                         "gender": rec["gender"], "split": split})
        print(f"  {shard.name}: {table.num_rows} clips", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / CLIPS_CSV, index=False)
    print(f"Wrote {len(frame)} clips, {frame['speaker_id'].nunique()} speakers, "
          f"to {out_dir / CLIPS_CSV}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        raise SystemExit("usage: python voxpopuli.py extract <shard dir> <out dir>")
    extract(sys.argv[2], sys.argv[3])
