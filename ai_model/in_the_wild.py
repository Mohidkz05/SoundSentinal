# in_the_wild.py
#
# Adapter for the In-the-Wild dataset (Müller et al., 2022) — 31,779 clips from
# 58 celebrities and politicians, real and deepfaked, collected from social
# networks and video platforms.
#
# WHY IT IS HERE. ASVspoof2019's nineteen attacks are from 2019. They predate
# neural codec models and every current commercial voice-cloning service, so a
# model scoring 3% EER on LA eval has been measured only against synthesis
# techniques that are now historical. In-the-Wild holds deepfakes people
# actually made and posted. The gap between the two numbers is the result.
#
# EVALUATION ONLY. This is never trained on and never selected on. Two reasons,
# both load-bearing:
#
#   1. The LA rows in APPROACH.md's comparison table are comparable to published
#      work precisely because the training set is unchanged. Training on this
#      would silently make every one of those comparisons invalid.
#   2. In-the-Wild is CC-BY-SA-4.0, and the clips underneath are scraped
#      recordings of real named people. Evaluating creates no derivative work
#      and ships nothing; training arguably does. See the licence discussion in
#      APPROACH.md.
#
# WHAT IT CANNOT MEASURE. No min t-DCF: that needs the organisers' ASV scores,
# which only ship with ASVspoof, and a t-DCF computed against a different ASV
# is not the same quantity. No per-attack breakdown either — these are
# real-world fakes with no attack taxonomy. EER is the whole result, and the
# clips carry a speaker label instead, which is kept in the JSON so a
# per-speaker analysis needs no re-run.

import os
from pathlib import Path

import pandas as pd

# Upstream writes "bona-fide"; ASVspoof and LABEL_MAP in model.py write
# "bonafide". One hyphen, and without this line every real clip is an unmapped
# key. Normalising here rather than widening LABEL_MAP keeps one spelling of
# the label inside the project.
LABEL_ALIASES = {"bona-fide": "bonafide", "bonafide": "bonafide", "spoof": "spoof"}

# Laid down by hpc/get_in_the_wild.slurm, which extracts the archive's own
# top-level release_in_the_wild/ directory into $ITW_ROOT.
RELEASE_DIR = "release_in_the_wild"
META_FILE = "meta.csv"


def get_itw_root():
    """The extracted dataset directory, from $ITW_ROOT.

    Falls back to an in-repo data/in_the_wild so a laptop checkout behaves the
    same way the corpus paths do, even though nothing here is in the repo.
    """
    env = os.getenv("ITW_ROOT")
    if env:
        root = Path(env)
    else:
        here = Path(__file__).resolve().parent
        root = here.parent / "data" / "in_the_wild"

    # Tolerate being pointed either at $ITW_ROOT or straight at the release
    # directory inside it, because both are reasonable readings of "the
    # dataset directory" and the difference is invisible until a file is missing.
    if (root / RELEASE_DIR / META_FILE).exists():
        return root / RELEASE_DIR
    if (root / META_FILE).exists():
        return root
    raise FileNotFoundError(
        f"No {META_FILE} under {root} or {root / RELEASE_DIR}.\n"
        f"Fetch it first:  sbatch hpc/get_in_the_wild.slurm\n"
        f"$ITW_ROOT is currently {env or '(unset)'}."
    )


def load_protocol(root=None):
    """meta.csv -> a protocol frame shaped like an ASVspoof one.

    Returns the same five columns AVSpoofDataset expects, so the dataset class,
    the preprocessing and the scoring loop are all shared with the ASVspoof
    path. Nothing about the model or the features is redefined here — see "The
    one rule" in CLAUDE.md.

    `audio_file_name` keeps its ".wav" extension and the dataset is given an
    empty suffix, rather than stripping the extension here and re-adding it
    there. One fewer place for the two to disagree.
    """
    root = Path(root) if root is not None else get_itw_root()
    meta = pd.read_csv(root / META_FILE)

    missing = {"file", "speaker", "label"} - set(meta.columns)
    if missing:
        raise ValueError(
            f"{root / META_FILE} is missing column(s) {sorted(missing)}; found "
            f"{list(meta.columns)}. The release layout may have changed — "
            f"re-run hpc/get_in_the_wild.slurm and read its layout report."
        )

    unknown = set(meta["label"]) - set(LABEL_ALIASES)
    if unknown:
        raise ValueError(
            f"Unrecognised label(s) {sorted(unknown)} in {root / META_FILE}. "
            f"Known: {sorted(LABEL_ALIASES)}."
        )

    return pd.DataFrame({
        "speaker_id": meta["speaker"],
        "audio_file_name": meta["file"],
        "_": "-",
        # No attack taxonomy exists for real-world deepfakes. "-" is what
        # ASVspoof itself uses for bonafide rows, and evaluate.py suppresses the
        # per-attack table when that is the only value.
        "system_id": "-",
        "label": meta["label"].map(LABEL_ALIASES),
    })
