# speechfake.py
#
# Adapter for SpeechFake (Huang et al., ACL 2025), bilingual subset "BD" —
# 704,862 training clips, English and Chinese, from 30 open-source TTS,
# voice-conversion and vocoder systems.
#
# WHY IT IS HERE. SSL-AASIST + RawBoost trained on ASVspoof2019 LA alone scores
# 11.21% EER on In-the-Wild (RESULTS.md Finding 8). A 2026 comparison of
# anti-spoofing training sets (arXiv 2606.08038) found that the number of
# distinct generators matters more than hours of audio, and put SpeechFake-BD
# at 2.63% on In-the-Wild against 17.20% for ASVspoof 5 — which is consistent
# with our Finding 7, where adding ASVspoof 5 did not help. `--extra-train
# speechfake` adds its baseline training split to LA train.
#
# WHAT IT DOES NOT FIX. Its bona fide audio is AISHELL-1/3, LibriTTS and VCTK:
# read speech, mostly clean. Common Voice ships with the multilingual subset,
# not this one. So this adds generator diversity, not real-world recording
# conditions — In-the-Wild's noisy bona fide is still unrepresented.
#
# TWO THINGS DIFFER FROM asvspoof5.py.
#
#   - Model selection uses LA dev PLUS SpeechFake dev (train_dp_avspoof.py).
#     SSL-AASIST reaches 0.00% on LA dev within a few epochs, after which every
#     epoch ties and best.pth is simply the first to tie; LA dev can no longer
#     choose. SpeechFake dev is 117k clips from the same 30 generators, so it
#     still discriminates.
#   - LA eval stops being a clean held-out test for this model. VCTK is the
#     source corpus of ASVspoof2019's bona fide speech and SpeechFake trains on
#     VCTK recordings and on fakes made from them. Report the LA eval number
#     with that caveat; In-the-Wild is unaffected, since nothing here touches it.
#
# In-the-Wild stays evaluation-only. Nothing in this file reads it.
#
# LICENCE. CC BY 4.0 (the repository's LICENSE.txt, checked 26 September 2026);
# bona fide sources are CC BY 4.0 (VCTK, LibriTTS) and Apache 2.0 (AISHELL-1/3).
# Cite Huang et al. (2025), arXiv:2507.21463. Do not re-host the audio.
#
# Laid down by hpc/get_speechfake.slurm.

import os
from pathlib import Path

import pandas as pd

# The authors' "baseline" protocol: train/dev/test drawn from the same 30
# generators. The cross_* protocols hold generators or languages out and are
# for evaluation designs we are not running.
PARTITIONS = ("train", "dev", "test")
COLUMNS = ["file", "label", "generator", "model", "speaker", "language"]


def get_speechfake_root():
    """The extracted corpus, from $SPEECHFAKE_ROOT, else an in-repo data/speechfake."""
    env = os.getenv("SPEECHFAKE_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "speechfake"
    if not (root / "BD").is_dir() or not (root / "Real").is_dir():
        raise FileNotFoundError(
            f"No BD/ and Real/ under {root}.\n"
            f"Fetch it first:  sbatch hpc/get_speechfake.slurm\n"
            f"$SPEECHFAKE_ROOT is currently {env or '(unset)'}."
        )
    return root


def load_protocol(partition="train", root=None):
    """Returns (protocol frame, audio root) for AVSpoofDataset, with suffix="".

    The frame has the same five columns an ASVspoof2019 protocol does, so the
    dataset class, the preprocessing and the class weighting are shared rather
    than redefined — see "The one rule" in CLAUDE.md. `file` is a path relative
    to the corpus root and already carries .wav, so the caller passes suffix="",
    as in_the_wild.py does. Sample rates are mixed (16, 24 and 48 kHz);
    preprocess_waveform resamples every clip to 16 kHz.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    root = Path(root) if root is not None else get_speechfake_root()
    proto = root / "metadata" / "experiments" / "baseline" / f"{partition}_all.csv"
    if not proto.is_file():
        raise FileNotFoundError(f"No {proto}. Was metadata.zip extracted?")

    meta = pd.read_csv(proto, dtype=str)
    if list(meta.columns) != COLUMNS:
        raise ValueError(
            f"{proto} has columns {list(meta.columns)}, expected {COLUMNS}. "
            f"The release format may have changed.")

    unknown = set(meta["label"]) - {"bonafide", "spoof"}
    if unknown:
        raise ValueError(f"Unrecognised label(s) {sorted(unknown)} in {proto}.")

    frame = pd.DataFrame({
        "speaker_id": meta["speaker"],
        "audio_file_name": meta["file"],
        "_": "-",
        # ASVspoof2019 writes "-" for bona fide rows; keep one convention.
        # `model` names the system (BigVGAN, CosyVoice, ...); `generator` is
        # only its family (NV/TTS/VC), too coarse to break results down by.
        "system_id": meta["model"].where(meta["label"] == "spoof", "-"),
        "label": meta["label"],
    })
    return frame, root
