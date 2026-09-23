# asvspoof5.py
#
# Adapter for ASVspoof 5 (Wang et al., 2024) — 182,357 training clips from
# crowdsourced speakers (Multilingual LibriSpeech) and newer TTS/VC attacks.
#
# WHY IT IS HERE. AASIST trained on ASVspoof2019 LA alone falls from 3.17% EER
# on LA eval to 37.15% on In-the-Wild (RESULTS.md Finding 6). LA's bona fide
# audio is one clean studio corpus and its attacks are from 2019, so a model can
# score well on it by learning "clean VCTK = real". ASVspoof 5's bona fide audio
# comes from ~2,000 speakers in diverse acoustic conditions. `--extra-train
# asvspoof5` adds its training partition to LA train; that is the whole use.
#
# WHAT THIS DOES NOT CHANGE.
#
#   - The LA rows. A model trained with extra data is a NEW row in the table.
#     Its checkpoint directory is separate (get_ckpt_paths), and every existing
#     number stays comparable to published work.
#   - Model selection and threshold calibration still use LA dev, as for every
#     other row, so the only variable is the training data.
#   - In-the-Wild stays evaluation-only. Nothing in this file reads it, and the
#     ITW number for a model trained with this data is legitimate precisely
#     because neither training corpus touches ITW.
#
# LICENCE. The database is ODC-By and its bona fide audio CC BY 4.0 — the same
# terms as ASVspoof2019, checked against the Zenodo record's LICENSE.txt on
# 23 September 2026. Attribution binds the writeup: cite Wang et al. (2024),
# doi:10.21437/ASVspoof.2024-1. Do not re-host the audio.
#
# Laid down by hpc/get_asvspoof5.slurm.

import os
from pathlib import Path

import pandas as pd

# partition -> (audio directory, protocol file), as the record's README names
# them. Only train is used for training; dev is here so it can be scored.
PARTITIONS = {
    "train": ("flac_T", "ASVspoof5.train.tsv"),
    "dev":   ("flac_D", "ASVspoof5.dev.track_1.tsv"),
}

# README section 3. Documented as "FIVE columns" and then listing ten; the list
# is what the files hold, so the count is checked rather than assumed.
COLUMNS = ["speaker_id", "file", "gender", "codec", "codec_q", "codec_seed",
           "attack_tag", "attack_label", "key", "tmp"]


def get_asv5_root():
    """The extracted corpus, from $ASV5_ROOT, else an in-repo data/asvspoof5."""
    env = os.getenv("ASV5_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "asvspoof5"
    if not (root / "flac_T").is_dir():
        raise FileNotFoundError(
            f"No flac_T/ under {root}.\n"
            f"Fetch it first:  sbatch hpc/get_asvspoof5.slurm\n"
            f"$ASV5_ROOT is currently {env or '(unset)'}."
        )
    return root


def find_protocol(root, name):
    """The protocol tarball's internal layout is not documented, so search for it."""
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"No {name} under {root}. Was ASVspoof5_protocols.tar.gz extracted?")
    return hits[0]


def load_protocol(partition="train", root=None):
    """Returns (protocol frame, audio directory) for AVSpoofDataset.

    The frame has the same five columns an ASVspoof2019 protocol does, so the
    dataset class, the preprocessing and the class weighting are shared rather
    than redefined — see "The one rule" in CLAUDE.md. Filenames carry no
    extension, matching ASVspoof2019, so the dataset's default ".flac" suffix
    applies unchanged.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(PARTITIONS)}, got {partition!r}")
    root = Path(root) if root is not None else get_asv5_root()
    audio_sub, proto_name = PARTITIONS[partition]
    proto = find_protocol(root, proto_name)

    meta = pd.read_csv(proto, sep=r"\s+", header=None, engine="python")
    if meta.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{proto} has {meta.shape[1]} columns, expected {len(COLUMNS)} "
            f"({' '.join(COLUMNS)}). The release format may have changed."
        )
    meta.columns = COLUMNS

    unknown = set(meta["key"]) - {"bonafide", "spoof"}
    if unknown:
        raise ValueError(f"Unrecognised key(s) {sorted(unknown)} in {proto}.")

    frame = pd.DataFrame({
        "speaker_id": meta["speaker_id"],
        "audio_file_name": meta["file"],
        "_": "-",
        # ASVspoof2019 writes "-" for bona fide rows; keep one convention.
        "system_id": meta["attack_label"].where(meta["key"] == "spoof", "-"),
        "label": meta["key"],
    })
    return frame, root / audio_sub
