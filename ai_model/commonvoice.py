# commonvoice.py
#
# Common Voice (Mozilla): volunteers reading sentences on their own
# microphones, in their own rooms. CC0. Shipped inside SpeechFake as
# Real/CommonVoice.zip (the bona fide half of its multilingual subset MD), so
# it is laid down by `sbatch --export=ALL,WITH_COMMONVOICE=1
# hpc/get_speechfake.slurm` and lives under $SPEECHFAKE_ROOT.
#
# TWO USES, KEPT APART BY SPLIT.
#
#   - TRAINING (`--extra-bonafide commonvoice`): the English TRAIN split,
#     33,614 clips, labelled bona fide. The LA + SpeechFake model ranks
#     In-the-Wild well (2.65% EER) but scores genuine real-world recordings as
#     suspicious, because every bona fide clip it trained on was clean read
#     speech (RESULTS.md Finding 9). Recalibrating the threshold could not fix
#     that; teaching it that home-recorded real speech is real might.
#   - CALIBRATION (calibrate.py): the English TEST split, 16,386 clips. For a
#     model trained with the train split, calibrate.py refuses anything else.
#
# The split is Common Voice's own, read off the path
# (Real/CommonVoice/en/<train|test>/...). Common Voice assigns each speaker to
# exactly one of train/dev/test, so calibration voices are voices the model
# never trained on. SpeechFake's metadata carries no speaker ids, so this
# cannot be re-checked here; it rests on Common Voice's published splitting.
#
# CAVEAT. XLS-R was pretrained on unlabelled Common Voice audio (voxpopuli.py
# has the same caveat). Training on it labelled is new; hearing it is not.
#
# In-the-Wild stays evaluation-only. Nothing in this file reads it.

from pathlib import Path

import pandas as pd

from speechfake import get_speechfake_root

METADATA = Path("metadata") / "Real" / "CommonVoice.csv"
SPLITS = ("train", "test")


def load_clips(language="en", split=None, root=None):
    """(frame with file/speaker_id/split, root). split=None returns both splits."""
    root = Path(root) if root is not None else get_speechfake_root()
    meta = pd.read_csv(root / METADATA, dtype=str)
    meta = meta[(meta["language"] == language) & (meta["label"] == "bonafide")]
    # Real/CommonVoice/<language>/<split>/<shard>/<clip>.wav
    parts = meta["file"].str.split("/")
    clips = pd.DataFrame({"file": meta["file"].values, "speaker_id": "-",
                          "split": parts.str[3].values})
    unknown = set(clips["split"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unexpected Common Voice split(s) {sorted(unknown)} in "
                         f"{root / METADATA}; the path layout may have changed.")
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        clips = clips[clips["split"] == split].reset_index(drop=True)
    missing = [f for f in clips["file"].head(20) if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root / missing[0]} does not exist. Was Real/CommonVoice.zip extracted?\n"
            f"Fetch it:  sbatch --export=ALL,WITH_COMMONVOICE=1 hpc/get_speechfake.slurm")
    return clips, root


def load_protocol(split="train", language="en", root=None):
    """(protocol frame, audio root) for AVSpoofDataset with suffix="": every row bona fide."""
    clips, root = load_clips(language, split, root)
    frame = pd.DataFrame({
        "speaker_id": clips["speaker_id"].values,
        "audio_file_name": clips["file"].values,
        "_": "-",
        "system_id": "-",
        "label": "bonafide",
    })
    return frame, root
