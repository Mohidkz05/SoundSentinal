# librispeech.py
#
# LibriSpeech test-clean (Panayotov et al., ICASSP 2015): 2,620 utterances of
# read audiobook speech by 40 speakers, recorded by LibriVox volunteers and
# selected for low noise. CC BY 4.0. Fetch with hpc/get_librispeech.slurm.
#
# WHY IT IS HERE. It is the CALIBRATION set for the clean-audio threshold
# (RESULTS.md Finding 15), and is never training data. Finding 13 found clean
# audio sits ~14 log-odds lower on the served model's scale than the noisy
# real speech its threshold was fitted on (People's Speech), so a third of
# clean fakes pass. A second threshold for clean recordings has to be fitted
# on clean REAL speech the model never trained on, and this is that.
#
# HELD OUT? The model's clean real speech in training is VCTK and LibriTTS
# train-clean-100 (via SpeechFake). LibriTTS is cut from LibriSpeech, but
# LibriSpeech's splits share no speaker, so test-clean's 40 speakers are not
# train-clean-100's 247. `check` verifies that against the speakers SpeechFake
# actually uses, rather than trusting it. Not held out: XLS-R's pretraining
# includes Multilingual LibriSpeech, which is LibriVox audio too and may
# contain the same recordings — heard without labels, but heard. Disclosed in
# Finding 15.
#
#     python librispeech.py check <root>   # speaker overlap with SpeechFake

import os
import sys
from pathlib import Path

import pandas as pd


def get_librispeech_root():
    """The unpacked corpus, from $LIBRISPEECH_ROOT, else an in-repo data/librispeech."""
    env = os.getenv("LIBRISPEECH_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "librispeech"
    if not (root / "LibriSpeech" / "test-clean").is_dir():
        raise FileNotFoundError(
            f"No LibriSpeech/test-clean under {root}.\n"
            f"Fetch it first:  sbatch hpc/get_librispeech.slurm\n"
            f"$LIBRISPEECH_ROOT is currently {env or '(unset)'}.")
    return root


def load_clips(root=None):
    """(frame of file, speaker_id), audio root. `file` is relative and carries .flac."""
    root = Path(root) if root is not None else get_librispeech_root()
    base = root / "LibriSpeech" / "test-clean"
    files = sorted(base.glob("*/*/*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files under {base}.")
    clips = pd.DataFrame({
        "file": [str(f.relative_to(root)) for f in files],
        "speaker_id": [f"LS{f.parent.parent.name}" for f in files],
    })
    return clips, root


def speechfake_libritts_speakers():
    """Every LibriTTS speaker id SpeechFake's baseline protocols use."""
    import speechfake
    sf_root = speechfake.get_speechfake_root()
    speakers = set()
    for part in speechfake.PARTITIONS:
        meta = speechfake.read_metadata(part, sf_root)
        lt = meta["speaker"][meta["speaker"].str.startswith("LT", na=False)]
        speakers |= set(lt.str[2:])
    return speakers


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        sys.exit("usage: python librispeech.py check <root>")
    clips, _ = load_clips(sys.argv[2])
    ours = set(clips["speaker_id"].str[2:])
    overlap = ours & speechfake_libritts_speakers()
    print(f"{len(clips)} utterances, {len(ours)} speakers")
    if overlap:
        sys.exit(f"ERROR: {len(overlap)} speaker(s) also in SpeechFake: {sorted(overlap)}. "
                 f"Not held out; do not calibrate on it.")
    print("No speaker in common with SpeechFake's LibriTTS speakers: held out.")
