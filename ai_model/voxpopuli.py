# voxpopuli.py
#
# VoxPopuli English (Wang et al., ACL 2021): speeches in the European
# Parliament, 2009-2020, recorded through the chamber's microphones and
# broadcast. CC0 (the European Parliament's legal notice also authorises
# commercial reuse of its recordings, with the source acknowledged).
# None of In-the-Wild's speakers sat in the European Parliament (checked
# against its meta.csv on 27 September 2026).
#
# TWO USES, KEPT APART.
#
#   - CALIBRATION (calibrate.py, Finding 9): the three shards fetched first —
#     en validation, en test and train shard 00000, 9,678 clips. Selected with
#     split="calibration", which is what calibrate.py's default reads, so the
#     Finding 9 and 10 numbers reproduce after more shards are added.
#   - TRAINING (`--extra-bonafide commonvoice+voxpopuli`, Finding 11): every
#     fetched TRAIN shard (00000-00005, ~36k clips), labelled bona fide. Real
#     speech through broadcast microphones, next to Common Voice's home
#     microphones, so that no single recording setup is what "real" means to
#     the model. Finding 10 showed what one setup does: the model learnt
#     Common Voice's conditions, not real speech.
#   - For a model trained on it, split="heldout" gives the validation and test
#     clips whose speakers are in no fetched train shard. VoxPopuli's own
#     splits share speakers (76 of train's first-shard speakers are also in
#     test), so the split name alone is not enough. calibrate.py uses this
#     only for --measure-only diagnostics, never to set a threshold.
#
# CAVEAT. XLS-R, the model's front-end, was pretrained on unlabelled
# VoxPopuli audio (as it was on Common Voice). Its features have heard this
# kind of audio before, which may make it score as more "familiar" than truly
# unseen real speech.
#
#     python voxpopuli.py extract <dir with the .parquet shards> <out dir>
#
# writes one audio file per clip and a clips.csv (file, speaker_id, gender,
# split, shard). Re-running it after adding shards skips files already on
# disk. Fetch the shards with hpc/get_voxpopuli.slurm.

import os
import sys
from pathlib import Path

import pandas as pd

CLIPS_CSV = "clips.csv"
# The shards calibrate.py sampled in Findings 9 and 10.
CALIBRATION_SHARDS = ("validation-00000-of-00001", "test-00000-of-00001",
                      "train-00000-of-00030")
SPLITS = ("calibration", "train", "heldout")


def get_voxpopuli_root():
    env = os.getenv("VOXPOPULI_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "voxpopuli"
    if not (root / CLIPS_CSV).is_file():
        raise FileNotFoundError(
            f"No {CLIPS_CSV} under {root}.\n"
            f"Fetch and extract it first:  sbatch hpc/get_voxpopuli.slurm\n"
            f"$VOXPOPULI_ROOT is currently {env or '(unset)'}.")
    return root


def load_clips(split="calibration", root=None):
    """(frame with file + speaker_id, root). Paths in `file` are relative to root."""
    root = Path(root) if root is not None else get_voxpopuli_root()
    clips = pd.read_csv(root / CLIPS_CSV, dtype=str)
    if "shard" not in clips.columns:
        raise ValueError(f"{root / CLIPS_CSV} predates the shard column. Re-extract: "
                         f"sbatch hpc/get_voxpopuli.slurm")
    if split == "calibration":
        clips = clips[clips["shard"].isin(CALIBRATION_SHARDS)]
    elif split == "train":
        clips = clips[clips["split"] == "train"]
    elif split == "heldout":
        train_speakers = set(clips.loc[clips["split"] == "train", "speaker_id"]) - {"-"}
        clips = clips[clips["split"].isin(["validation", "test"])
                      & ~clips["speaker_id"].isin(train_speakers)
                      & (clips["speaker_id"] != "-")]
    else:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return clips.reset_index(drop=True), root


def load_protocol(split="train", root=None):
    """(protocol frame, audio root) for AVSpoofDataset with suffix="": every row bona fide."""
    clips, root = load_clips(split, root)
    frame = pd.DataFrame({
        "speaker_id": clips["speaker_id"].values,
        "audio_file_name": clips["file"].values,
        "_": "-",
        "system_id": "-",
        "label": "bonafide",
    })
    return frame, root


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
        written = 0
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=256, columns=["audio_id", "audio", "speaker_id", "gender"]):
            for rec in batch.to_pylist():
                audio = rec["audio"]
                ext = Path(audio.get("path") or "x.wav").suffix or ".wav"
                rel = Path("audio") / split / f"{rec['audio_id']}{ext}"
                target = out_dir / rel
                if not (target.is_file() and target.stat().st_size == len(audio["bytes"])):
                    target.write_bytes(audio["bytes"])
                    written += 1
                rows.append({"file": str(rel), "speaker_id": rec["speaker_id"] or "-",
                             "gender": rec["gender"], "split": split,
                             "shard": shard.name.removesuffix(".parquet")})
        print(f"  {shard.name}: {sum(r['shard'] == shard.stem for r in rows)} clips, "
              f"{written} newly written", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / CLIPS_CSV, index=False)
    print(f"Wrote {len(frame)} clips, {frame['speaker_id'].nunique()} speakers, "
          f"to {out_dir / CLIPS_CSV}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        raise SystemExit("usage: python voxpopuli.py extract <shard dir> <out dir>")
    extract(sys.argv[2], sys.argv[3])
