# mlaad.py
#
# Adapter for MLAAD (Müller et al., "MLAAD: The Multi-Language Audio
# Anti-Spoofing Dataset"), English only: 143 text-to-speech systems, about
# 1,000 fake clips each, from 2019-era Coqui models to ElevenLabs v3, OpenAI
# TTS, Cartesia, MiniMax, Gemini and VibeVoice. Fakes only; it ships no real
# speech of its own.
#
# WHY IT IS HERE (RESULTS.md Finding 17). Finding 9's jump to 2.65% on
# In-the-Wild came from generator diversity — 30 systems instead of LA's six —
# not from hours of audio. MLAAD is the same lever again, and its systems are
# the modern commercial cloning users actually worry about, which neither LA
# nor SpeechFake has.
#
# THE SPLIT IS BY SYSTEM, AND FIXED HERE, before anything is scored. A system's
# clips are all in one split, so the held-out splits measure generators the
# model has never heard:
#
#   train       everything not held out, including every system whose family
#               SpeechFake already trains on (SPEECHFAKE_FAMILIES) — holding
#               those out would not test an unseen generator
#   heldout-a   >= 20 systems: Stage A (choosing, going on). Never trained on.
#   heldout-b   >= 20 more: Stage B (the test). Never trained or chosen on.
#
# Held-out is drawn by model FAMILY (family()), so no sibling version of a
# held-out system trains: families in numpy RandomState(42) order until each
# held-out split has at least 20 systems;
# SYSTEMS is the snapshot at REVISION, so a later upload cannot move a system
# between splits. Real speech for the held-out splits' EER is LibriSpeech
# test-clean, split by speaker (librispeech.py): 20 speakers to each, never
# trained on. It is audiobook speech, like MLAAD's text source.
#
# LICENCE. CC BY-NC 4.0 — non-commercial. Fine for this university project; a
# model trained on it cannot be sold. Gated on Hugging Face: each user accepts
# the terms once, then hpc/get_mlaad.slurm downloads with their token.
#
#     python mlaad.py split          # print the three splits
#     python mlaad.py check <root>   # every system present, clip counts

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "mueller91/MLAAD"
REVISION = "30c3dec763fa4803f11c2df1557d3b0828395366"   # 30 September 2026
LANGUAGE = "en"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")
SPLITS = ("train", "heldout-a", "heldout-b")
N_HELDOUT = 20
SEED = 42

SYSTEMS = [
    'Audio8-TTS-Preview-0.1b', 'Audio8-TTS-Preview-0.6b', 'Cartesia.ai (Sonic-3)',
    'ChatTTS', 'Chatterbox', 'Chatterbox-Turbo', 'DeepGram', 'Dramabox', 'Echo-TTS',
    'Edge-TTS', 'ElevenLabs-Turbo-v2.5', 'ElevenLabs-v2-Multilingual', 'ElevenLabs-v3',
    'FireRedTTS-2.0', 'Fish-S2-Pro', 'FishTTS', 'GLM-TTS', 'GPA-v1.0', 'GPA-v1.5',
    'GPT-SoVITS', 'Gemini-3.1-Flash-TTS', 'Higgs-Audio-V2', 'Higgs-Audio-V3',
    'Hume TADA-3B-ML', 'Index-TTS-1.5', 'Index-TTS-2.0', 'Indri-TTS-0.1',
    'Inflect-Micro-v2', 'Inflect-Nano-v1', 'Inflect-Nano-v2', 'Inworld-TTS-2',
    'Kani-TTS-370M', 'Kitten-TTS-Nano-0.1', 'Kitten-TTS-Nano-0.2', 'KugelAudio',
    'Kyutai-TTS', 'LEMAS-TTS', 'LFM2.5-Audio', 'Llasa-1B', 'Llasa-1B-Multilingual',
    'Llasa-3B', 'Llasa-8B', 'LongCat-AudioDiT', 'LuxTTS', 'MOSS-TTS-1.7B',
    'MOSS-TTS-8B', 'MOSS-TTS-Nano-100M', 'Mars5', 'Marvis-TTS', 'MatchaTTS',
    'Maya1 TTS', 'MegaTTS3', 'MeloTTS', 'Metavoice-1B', 'Microsoft VibeVoice 1.5B',
    'Microsoft VibeVoice Large', 'MingTTS', 'MiniCPM-o-2.6',
    'MiniMax-Speech-2.8-Turbo', 'MiraTTS', 'MisoTTS', 'Nari Dia-1.6B', 'Nari Dia2',
    'NeuTTS-Air', 'NeuTTS-Nano', 'OmniVoice', 'OpenAI TTS-1 HD', 'OpenVoiceV2',
    'Openaudio-S1-Mini', 'OuteTTS', 'PocketTTS', 'PrimeTTS', 'Qwen2.5-Omni',
    'Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base', 'Qwen3-TTS-CustomVoice',
    'RVC', 'Raon-OpenTTS', 'Resemble.ai (April 12th, 2025)', 'Rime-Coda',
    'Ringg Squirrel TTS v1.0', 'Smallest-Lightning-v3.1', 'Soprano11', 'SopranoTTS',
    'Sopro', 'SoulX-Podcast', 'Spark-TTS-0.5B', 'Step-Audio-EditX', 'Supertonic',
    'TADA-1B', 'Veena', 'VibeVoice-Realtime-0.5B', 'VoXtream2', 'VoxCPM-0.5B',
    'VoxCPM-1.5', 'VoxCPM2', 'Voxtral', 'Voxtream', 'WavTTS', 'WhisperSpeech',
    'ZONOS2', 'ZipVoice', 'dots.tts-base', 'dots.tts-mf', 'dots.tts-soar', 'e2-tts',
    'f5-tts', 'facebook_mms-tts-eng', 'griffin_lim', 'kokoro',
    'microsoft_speecht5_tts', 'minimax_speech-02-turbo', 'minimax_speech-2.6-hd',
    'optispeech', 'orpheus-tts-0.1-finetune', 'parler_tts_large_v1',
    'parler_tts_mini_v0.1', 'parler_tts_mini_v1', 'sarashina2.2-tts', 'sesame_csm',
    'suno_bark', 'suno_bark-small', 'supertonic-3',
    'tts_models_en_blizzard2013_capacitron-t2-c50', 'tts_models_en_ek1_tacotron2',
    'tts_models_en_jenny_jenny', 'tts_models_en_ljspeech_fast_pitch',
    'tts_models_en_ljspeech_glow-tts', 'tts_models_en_ljspeech_neural_hmm',
    'tts_models_en_ljspeech_overflow', 'tts_models_en_ljspeech_speedy-speech',
    'tts_models_en_ljspeech_tacotron2-DCA', 'tts_models_en_ljspeech_tacotron2-DDC',
    'tts_models_en_ljspeech_tacotron2-DDC_ph', 'tts_models_en_ljspeech_vits',
    'tts_models_en_ljspeech_vits--neon', 'tts_models_en_multi-dataset_tortoise-v2',
    'tts_models_en_sam_tacotron-DDC', 'tts_models_multilingual_multi-dataset_bark',
    'tts_models_multilingual_multi-dataset_xtts_v1.1',
    'tts_models_multilingual_multi-dataset_xtts_v2', 'vixTTS', 'zonosTTS-v0.1',
]

# Families SpeechFake's 30 training systems belong to, matched case-blind on
# the MLAAD folder name. These stay in train: a "held-out" XTTS clip would
# test a generator the model already knows.
SPEECHFAKE_FAMILIES = ("chattts", "fish", "openaudio", "gpt-sovits", "melotts", "parler",
                       "openvoice", "firered", "tortoise", "glow-tts", "tacotron",
                       "cosyvoice", "styletts", "xtts")


# Folder-name prefixes that say nothing about the model: the toolkit (Coqui's
# tts_models_*), the corpus a voice was trained on, or the vendor in front of
# a model name that appears elsewhere without it (Hume TADA = TADA).
_PREFIXES = ("tts_models_multilingual_multi-dataset_", "tts_models_en_ljspeech_",
             "tts_models_en_", "microsoft ", "microsoft_", "facebook_", "suno_", "hume ")


def family(system):
    """The model family a system belongs to: its first name token, less a
    trailing "tts" and version digits. Llasa-1B and Llasa-8B are one family,
    as are VibeVoice-Realtime and Microsoft VibeVoice Large, suno_bark and the
    multilingual bark. Deliberately coarse: grouping two different models
    only makes held-out stricter, never leakier."""
    import re
    name = system.lower()
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    token = re.split(r"[-_. (]", name)[0]
    return re.sub(r"(tts)?[0-9]*$", "", token) or token


def splits():
    """{split: [system, ...]}, deterministic from SYSTEMS and SEED.

    Split by family, not by system: holding out Llasa-8B while training on
    Llasa-1B would test a sibling, not an unseen generator. Families are taken
    in a seeded random order into heldout-a until it has N_HELDOUT systems,
    then into heldout-b likewise; every other family trains."""
    eligible = sorted({family(s) for s in SYSTEMS
                       if not any(f in s.lower() for f in SPEECHFAKE_FAMILIES)})
    members = {f: sorted(s for s in SYSTEMS if family(s) == f) for f in eligible}
    # A family with a SpeechFake-overlapping member stays in train whole.
    eligible = [f for f in eligible
                if not any(k in s.lower() for s in members[f] for k in SPEECHFAKE_FAMILIES)]
    out = {"heldout-a": [], "heldout-b": []}
    for i in np.random.RandomState(SEED).permutation(len(eligible)):
        for split in out:
            if len(out[split]) < N_HELDOUT:
                out[split] += members[eligible[i]]
                break
    held = set(out["heldout-a"]) | set(out["heldout-b"])
    return {"train": sorted(s for s in SYSTEMS if s not in held),
            "heldout-a": sorted(out["heldout-a"]), "heldout-b": sorted(out["heldout-b"])}


def get_mlaad_root():
    env = os.getenv("MLAAD_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "mlaad"
    if not (root / "fake" / LANGUAGE).is_dir():
        raise FileNotFoundError(
            f"No fake/{LANGUAGE}/ under {root}.\n"
            f"Fetch it first:  sbatch hpc/get_mlaad.slurm\n"
            f"$MLAAD_ROOT is currently {env or '(unset)'}.")
    return root


def _files(root, system):
    d = Path(root) / "fake" / LANGUAGE / system
    return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)


def load_protocol(split="train", root=None):
    """(protocol frame, audio root) for AVSpoofDataset with suffix="": the
    split's fakes, labelled spoof, system_id the MLAAD system name."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(root) if root is not None else get_mlaad_root()
    rows = [(system, str(p.relative_to(root)))
            for system in splits()[split] for p in _files(root, system)]
    frame = pd.DataFrame({
        "speaker_id": [s for s, _ in rows],
        "audio_file_name": [f for _, f in rows],
        "_": "-",
        "system_id": [s for s, _ in rows],
        "label": "spoof",
    })
    return frame, root


def load_heldout(split, mlaad_root=None, librispeech_root=None):
    """A held-out split's fakes plus its half of LibriSpeech test-clean as the
    real side, as one frame whose paths are relative to a common parent.
    heldout-a takes the first 20 speakers in sorted order, heldout-b the rest."""
    import librispeech
    if split not in ("heldout-a", "heldout-b"):
        raise ValueError(f"load_heldout takes a held-out split, got {split!r}")
    fakes, m_root = load_protocol(split, mlaad_root)
    clips, l_root = librispeech.load_clips(librispeech_root)
    speakers = sorted(clips["speaker_id"].unique())
    half = set(speakers[:len(speakers) // 2] if split == "heldout-a"
               else speakers[len(speakers) // 2:])
    clips = clips[clips["speaker_id"].isin(half)]
    # One root for both: absolute paths joined onto "/" by AVSpoofDataset.
    fakes = fakes.assign(audio_file_name=[str(m_root / f) for f in fakes["audio_file_name"]])
    real = pd.DataFrame({
        "speaker_id": clips["speaker_id"].to_numpy(),
        "audio_file_name": [str(l_root / f) for f in clips["file"]],
        "_": "-", "system_id": "-", "label": "bonafide",
    })
    return pd.concat([real, fakes], ignore_index=True), Path("/")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "split":
        for name, systems in splits().items():
            print(f"{name} ({len(systems)}): {', '.join(systems)}\n")
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        root = Path(sys.argv[2])
        missing = [s for s in SYSTEMS if not (root / "fake" / LANGUAGE / s).is_dir()]
        if missing:
            sys.exit(f"ERROR: {len(missing)} system(s) missing: {missing}")
        for name in SPLITS:
            frame, _ = load_protocol(name, root)
            counts = frame["system_id"].value_counts()
            print(f"{name}: {len(frame)} clips, {counts.size} systems, "
                  f"{counts.min()}-{counts.max()} per system")
    else:
        sys.exit("usage: python mlaad.py split | check <root>")
