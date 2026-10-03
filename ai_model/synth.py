# synth.py
#
# SoundSentinal's own fakes: clips generated here, with open text-to-speech
# models whose code AND weights allow commercial use and download without an
# account (RESULTS.md Finding 18). Two jobs:
#
#   1. more generator diversity in training — Finding 9's lever — under terms
#      the project can use commercially (MLAAD could not be, Finding 17);
#   2. a test of generators the model has NEVER heard, split by family.
#
# FAMILIES is the registry, fixed before anything is generated: each family,
# its models (Hugging Face repo, licence), and how it picks a voice. The split
# is by family, drawn once with RandomState(42); Parler stays in train because
# SpeechFake already trains on it. `python synth.py split` prints it.
#
# VOICES AND TEXT. Every fake reads a LibriSpeech transcript (public-domain
# books; LibriSpeech is CC BY 4.0, OpenSLR 12, no account):
#
#   train       prompts and text from train-clean-100: its 251 speakers are
#               the LibriTTS speakers already in training as REAL speech
#               (via SpeechFake), so the model sees the same people real and
#               faked — the difference it can learn is the generator's
#   heldout-a   prompts and text from test-clean speakers 1-20 (sorted)
#   heldout-b   the other 20; each held-out split's real side is the same
#               speakers' genuine test-clean clips
#   probe-a     Finding 19's Stage 0 only: heldout-a's speakers' genuine
#               clips re-encoded through each codec. Diagnostic, never trained
#
# Voice-cloning models clone the prompt speaker; the rest use their own
# preset voices (Kokoro, Kitten, Piper, Soprano, SpeechT5's CMU ARCTIC
# x-vectors, Parler/Maya1 text descriptions, Kyutai's CC0 and VCTK voices).
# The codec "family" (Finding 19) is not a TTS model: it re-encodes the
# genuine utterance whose transcript the job names ("resynth" voices).
#
# Generated audio lives on scratch under $SYNTH_ROOT/<split>/<family>/<model>/
# with a manifest.csv per model; synth/ holds the generator scripts.
#
#     python synth.py split                 # the fixed split
#     python synth.py check <root>          # what has been generated

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_HELDOUT_FAMILIES = 4          # per held-out split
CLIPS_TRAIN = 6000              # per family, spread over its models
CLIPS_HELDOUT = 1000            # per family
SPLITS = ("train", "heldout-a", "heldout-b", "probe-a")
# Diagnostic splits: which held-out split's speakers they draw from.
PROBES = {"probe-a": "heldout-a"}

# family -> [(model id, Hugging Face repo, licence, voice source)]
# voice source: "clone" (LibriSpeech prompt), "preset" (the model's own voices),
# "resynth" (the genuine utterance itself, through a neural codec)
FAMILIES = {
    "kokoro":    [("kokoro-82m", "hexgrad/Kokoro-82M", "Apache-2.0", "preset")],
    "chatterbox": [("chatterbox", "ResembleAI/chatterbox", "MIT", "clone"),
                   ("chatterbox-turbo", "ResembleAI/chatterbox-turbo", "MIT", "clone")],
    "dia":       [("dia-1.6b", "nari-labs/Dia-1.6B-0626", "Apache-2.0", "clone")],
    "zonos":     [("zonos-transformer", "Zyphra/Zonos-v0.1-transformer", "Apache-2.0", "clone")],
    # The 1.5B voice-cloning model's code was withdrawn by Microsoft after
    # misuse (their README, 5 Sep 2025); the supported streaming model is used.
    "vibevoice": [("vibevoice-realtime-0.5b", "microsoft/VibeVoice-Realtime-0.5B", "MIT",
                   "preset")],
    "parler":    [("parler-mini-v1", "parler-tts/parler-tts-mini-v1", "Apache-2.0", "preset")],
    "kyutai":    [("kyutai-tts-1.6b", "kyutai/tts-1.6b-en_fr", "CC-BY-4.0", "preset")],
    "kitten":    [("kitten-nano-0.2", "KittenML/kitten-tts-nano-0.2", "Apache-2.0", "preset")],
    "oute":      [("outetts-1.0-0.6b", "OuteAI/OuteTTS-1.0-0.6B", "Apache-2.0", "clone")],
    "speecht5":  [("speecht5", "microsoft/speecht5_tts", "MIT", "preset")],
    "piper":     [("piper-ljspeech", "rhasspy/piper-voices", "MIT; voice data public domain", "preset"),
                  ("piper-libritts-r", "rhasspy/piper-voices", "MIT; voice data CC BY 4.0", "preset"),
                  ("piper-cori", "rhasspy/piper-voices", "MIT; voice data public domain", "preset")],
    "qwen3tts":  [("qwen3-tts-0.6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Apache-2.0", "clone"),
                  ("qwen3-tts-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Apache-2.0", "clone")],
    "voxcpm":    [("voxcpm-0.5b", "openbmb/VoxCPM-0.5B", "Apache-2.0", "clone"),
                  ("voxcpm-1.5", "openbmb/VoxCPM1.5", "Apache-2.0", "clone")],
    "marvis":    [("marvis-250m", "Marvis-AI/marvis-tts-250m-v0.1-transformers", "Apache-2.0",
                   "clone")],
    "maya1":     [("maya1", "maya-research/maya1", "Apache-2.0", "preset")],
    "soprano":   [("soprano-80m", "ekwek/Soprano-80M", "Apache-2.0", "preset")],
    # Finding 19: genuine speech through open neural codecs, labelled spoof.
    # Never a codec a held-out family decodes with (DAC: Dia; Mimi: Marvis;
    # the Qwen3-TTS tokenizer; VoxCPM's audio VAE).
    "codec":     [("snac-24khz", "hubertsiuzdak/snac_24khz", "MIT", "resynth"),
                  ("wavtokenizer-75", "novateur/WavTokenizer-large-speech-75token", "MIT",
                   "resynth")],
}
# Never held out: Parler because SpeechFake already trains on it; codec
# because it is Finding 19's training data, added after the split was drawn —
# listing it here keeps the seeded split exactly as Finding 18 drew it.
ALWAYS_TRAIN = ("parler", "codec")


def splits():
    """{split: [family, ...]}: ALWAYS_TRAIN in train, the rest in a seeded
    order, N_HELDOUT_FAMILIES to heldout-a, the next to heldout-b."""
    pool = sorted(f for f in FAMILIES if f not in ALWAYS_TRAIN)
    order = [pool[i] for i in np.random.RandomState(SEED).permutation(len(pool))]
    a, b = order[:N_HELDOUT_FAMILIES], order[N_HELDOUT_FAMILIES:2 * N_HELDOUT_FAMILIES]
    return {"train": sorted(set(FAMILIES) - set(a) - set(b)),
            "heldout-a": sorted(a), "heldout-b": sorted(b), "probe-a": ["codec"]}


def split_of(family):
    """The split a family trains or is tested in (probes are extra, not its split)."""
    return next(s for s, fams in splits().items() if family in fams and s not in PROBES)


def get_synth_root():
    env = os.getenv("SYNTH_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "synth"
    if not root.is_dir():
        raise FileNotFoundError(f"No generated fakes at {root}. See synth/README.md.")
    return root


def load_manifests(split, root=None):
    """Every generated clip of a split: (frame of file, family, model, speaker,
    text), audio root. `file` is relative to the root."""
    root = Path(root) if root is not None else get_synth_root()
    frames = []
    for family in splits()[split]:
        for model, *_ in FAMILIES[family]:
            d = root / split / family / model
            m = d / "manifest.csv"
            if not m.is_file():
                continue
            # The pre-registered quality gate (Finding 18): a model is used
            # only once synth/qc.py has passed it. No report, or a failing
            # one, and its clips stay out — reported, never silently mixed in.
            qc = d / "qc.json"
            if not qc.is_file() or not json.loads(qc.read_text()).get("passes"):
                print(f"synth: skipping {split}/{family}/{model} "
                      f"({'no QC report' if not qc.is_file() else 'failed QC'})")
                continue
            frames.append(pd.read_csv(m, dtype=str).assign(family=family, model=model))
    if not frames:
        raise FileNotFoundError(f"No manifests for {split} under {root}.")
    return pd.concat(frames, ignore_index=True), root


def load_protocol(split="train", root=None):
    """(protocol frame, audio root) for AVSpoofDataset with suffix="": the
    split's fakes, labelled spoof, system_id the model id."""
    clips, root = load_manifests(split, root)
    frame = pd.DataFrame({
        "speaker_id": clips["speaker"], "audio_file_name": clips["file"], "_": "-",
        "system_id": clips["model"], "label": "spoof"})
    return frame, root


def load_heldout(split, synth_root=None, librispeech_root=None):
    """A held-out split's fakes plus the genuine test-clean clips of the same
    20 speakers, as one frame of absolute paths under root "/"."""
    import librispeech
    fakes, s_root = load_protocol(split, synth_root)
    clips, l_root = librispeech.load_clips(librispeech_root)
    real = clips[clips["speaker_id"].isin(heldout_speakers(split, clips))]
    fakes = fakes.assign(audio_file_name=[str(s_root / f) for f in fakes["audio_file_name"]])
    real = pd.DataFrame({
        "speaker_id": real["speaker_id"].to_numpy(),
        "audio_file_name": [str(l_root / f) for f in real["file"]],
        "_": "-", "system_id": "-", "label": "bonafide"})
    return pd.concat([real, fakes], ignore_index=True), Path("/")


def heldout_speakers(split, clips):
    """test-clean speaker ids for a held-out split: first or second half, sorted."""
    split = PROBES.get(split, split)
    speakers = sorted(clips["speaker_id"].unique())
    half = len(speakers) // 2
    return set(speakers[:half] if split == "heldout-a" else speakers[half:])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "split":
        for name, fams in splits().items():
            models = [m for f in fams for m, *_ in FAMILIES[f]]
            print(f"{name}: {', '.join(fams)}\n    models: {', '.join(models)}")
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        root = Path(sys.argv[2])
        for split in SPLITS:
            for family in splits()[split]:
                for model, *_ in FAMILIES[family]:
                    m = root / split / family / model / "manifest.csv"
                    n = len(pd.read_csv(m)) if m.is_file() else 0
                    print(f"{split:10s} {family:11s} {model:20s} {n:6d}")
    else:
        sys.exit("usage: python synth.py split | check <root>")
