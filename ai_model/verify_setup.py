# verify_setup.py
#
# Smoke test for the model/serving contract. Run this after changing anything in
# model.py or aasist.py, and after a fresh clone, before assuming the pipeline
# works:
#
#     python verify_setup.py
#
# It needs torch/torchaudio but does NOT need the ASVspoof dataset or trained
# weights — it uses the two sample .flac files committed alongside it.

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import (ARCH_FRONTENDS, ARCHITECTURES, DEFAULT_ARCH, DP_ARCHITECTURES,
                   DEFAULT_FRONTEND, FRONTENDS, MAX_LEN, N_LFCC, N_MELS,
                   DEGRADATIONS, SAMPLE_RATE, build_model, build_transform, check_pairing,
                   default_frontend_for, degrade_waveform, load_audio, preprocess_waveform)

from rawboost import ALGOS as RAWBOOST_ALGOS, RawBoost

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE = SCRIPT_DIR / "LA_T_1000137.flac"

# Published parameter counts, from Table 2 of the AASIST paper and the official
# repo. Ours differ by exactly the 512 dead bn1 weights that deviation 2 in
# aasist.py removes, so these are the numbers to expect rather than 297k/85k.
EXPECTED_PARAMS = {"cnn": 267_330, "aasist": 297_354, "aasist-l": 85_034,
                   # XLS-R 300M as transformers counts it (315,437,696) plus the
                   # 446,730-parameter AASIST head of ssl_aasist.py.
                   "ssl-aasist": 315_884_426}

failures = []


def check(name, condition, extra=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {extra}")
    if not condition:
        failures.append(name)


class _AugmentedCopies(torch.utils.data.Dataset):
    """The same clip, augmented on every read. Module-level so workers can pickle it."""

    def __init__(self, waveform, sample_rate, augment, n=4):
        self.waveform, self.sample_rate, self.augment, self.n = waveform, sample_rate, augment, n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.augment(self.waveform, self.sample_rate)


def main():
    waveform, sample_rate = load_audio(str(SAMPLE))

    # One shape per front-end. The raw front-end is the padded waveform itself:
    # AASIST's SincConv is the transform, so there is nothing to apply here.
    expected_shape = {
        "logmel": (1, N_MELS, 126),
        "lfcc": (1, N_LFCC * 3, 126),
        "raw": (1, MAX_LEN),
    }

    specs = {}
    for frontend in FRONTENDS:
        print(f"--- Preprocessing: {frontend} ---")
        spec = preprocess_waveform(waveform, sample_rate, build_transform(frontend))
        specs[frontend] = spec
        want = expected_shape[frontend]
        check(f"{frontend} shape is {want}", tuple(spec.shape) == want, tuple(spec.shape))
        check(f"{frontend} standardized to mean ~0",
              abs(float(spec.mean())) < 1e-4, f"mean={float(spec.mean()):.2e}")
        check(f"{frontend} standardized to std ~1",
              abs(float(spec.std()) - 1.0) < 1e-2, f"std={float(spec.std()):.4f}")

        # Every architecture that reads this front-end must accept the tensor it
        # produces. A mismatch here is the failure mode that matters: nothing
        # would crash at runtime, it would just be a confident wrong answer.
        for arch in ARCHITECTURES:
            if frontend not in ARCH_FRONTENDS[arch]:
                continue
            net = build_model(arch).eval()
            with torch.no_grad():
                out = net(spec.unsqueeze(0))
            check(f"{arch} on {frontend} returns 2 logits",
                  tuple(out.shape) == (1, 2), tuple(out.shape))

    check("the three front-ends differ",
          len({tuple(s.shape) for s in specs.values()}) == len(specs),
          " vs ".join(str(tuple(s.shape)) for s in specs.values()))

    print("--- Architectures ---")
    for arch in ARCHITECTURES:
        net = build_model(arch)
        n = sum(p.numel() for p in net.parameters() if p.requires_grad)
        check(f"{arch} has {EXPECTED_PARAMS[arch]:,} parameters",
              n == EXPECTED_PARAMS[arch], f"{n:,}")

        # The DP-compatibility invariant, and the reason aasist.py substitutes
        # GroupNorm throughout. BatchNorm mixes statistics across a batch, which
        # breaks DP-SGD's per-sample gradient guarantee — Opacus refuses to wrap
        # such a model at all. This must stay true of every architecture here,
        # whether or not the next run uses DP. See APPROACH.md.
        bn = [n for n, m in net.named_modules()
              if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        check(f"{arch} contains no BatchNorm (DP-compatible)", not bn, bn[:3])

    print("--- Differential privacy ---")
    # "No BatchNorm" is necessary but not sufficient. Opacus attributes every
    # gradient to the sample that produced it by hooking MODULES, so a free
    # nn.Parameter silently gets no per-sample gradient, and a free parameter on
    # the root module also drags the whole model into Opacus's trainable-layer-
    # with-buffers check. Both were true of upstream AASIST; deviation 8 in
    # aasist.py is what fixes them. Nothing about that is visible by reading the
    # model, so it is asserted here by actually taking one private step.
    #
    # This guards the project's central measurement. DP is off for the main
    # results (APPROACH.md), but the cost-of-privacy gap is the contribution,
    # and it is only worth reporting on an architecture worth using.
    try:
        from torch.utils.data import DataLoader, TensorDataset
        from opacus import PrivacyEngine

        # SSL-AASIST is excluded by design (DP_ARCHITECTURES in model.py).
        for arch in DP_ARCHITECTURES:
            frontend = default_frontend_for(arch)
            # A short clip: this is a wiring test, and AASIST's activations are
            # large enough that a full 4 seconds would dominate the runtime.
            shape = (1, 8000) if frontend == "raw" else (1, N_MELS, 126)
            net = build_model(arch)
            ds = TensorDataset(torch.randn(4, *shape), torch.randint(0, 2, (4,)))
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
            engine = PrivacyEngine()
            net, opt, loader = engine.make_private(
                module=net, optimizer=opt, data_loader=DataLoader(ds, batch_size=2),
                noise_multiplier=1.1, max_grad_norm=1.0)
            x, y = next(iter(loader))
            nn.CrossEntropyLoss()(net(x), y).backward()
            opt.step()
            check(f"{arch} takes a DP-SGD step under Opacus", True,
                  f"ε={engine.get_epsilon(delta=1e-5):.2f}")
    except ImportError:
        print("  [SKIP] opacus not installed")
    except Exception as e:
        check("every architecture takes a DP-SGD step under Opacus", False,
              f"{type(e).__name__}: {str(e)[:160]}")

    print("--- Architecture / front-end pairing ---")
    for arch in ARCHITECTURES:
        check(f"{arch} defaults to a front-end it can read",
              default_frontend_for(arch) in ARCH_FRONTENDS[arch],
              default_frontend_for(arch))
    try:
        check_pairing("aasist", "logmel")
        check("an impossible pairing is rejected", False, "no exception raised")
    except ValueError:
        check("an impossible pairing is rejected", True)

    print("--- The CNN's adaptive pool ---")
    net = build_model("cnn").eval()
    check("fc1 is Linear(2048, 128)", tuple(net.fc1.weight.shape) == (128, 2048),
          tuple(net.fc1.weight.shape))
    # The whole point of AdaptiveAvgPool2d: neither input dimension must matter.
    for channels, frames in ((N_MELS, 63), (N_MELS, 126), (N_MELS, 400), (N_LFCC * 3, 126)):
        with torch.no_grad():
            o = net(torch.randn(1, 1, channels, frames))
        check(f"handles input {channels}x{frames}", tuple(o.shape) == (1, 2), tuple(o.shape))

    print("--- AASIST is length-independent ---")
    # The 23 frequency nodes come from the 70 sinc filters, not from the clip
    # length, so a different MAX_LEN must not change the network's shape. This
    # is what lets us use 64000 samples where upstream uses 64600.
    aasist = build_model("aasist").eval()
    for samples in (MAX_LEN, 64600, 32000):
        with torch.no_grad():
            o = aasist(torch.randn(1, 1, samples))
        check(f"handles {samples} samples", tuple(o.shape) == (1, 2), tuple(o.shape))

    print("--- RawBoost augmentation ---")
    # Training-only (rawboost.py). Every algo must hand preprocess_waveform a
    # mono waveform of the same length, finite, and actually changed.
    for algo in sorted(RAWBOOST_ALGOS):
        out = RawBoost(algo)(waveform, sample_rate)
        same_len = tuple(out.shape) == (1, waveform.shape[1])
        ok = same_len and bool(torch.isfinite(out).all()) and not torch.equal(out, waveform[:1])
        check(f"algo {algo} ({RAWBOOST_ALGOS[algo]}) keeps shape, finite, changes the clip",
              ok, tuple(out.shape))
    # The claim rawboost.py's header rests on: numpy's global generator is
    # reseeded per DataLoader worker and per epoch. If it were not, every worker
    # would apply the same "random" distortion in lockstep and the augmentation
    # would be a fraction as varied as it looks.
    loader = torch.utils.data.DataLoader(
        _AugmentedCopies(waveform, sample_rate, RawBoost(5)), batch_size=1, num_workers=2)
    draws = [b[0] for _ in range(2) for b in loader]
    distinct = len({d.numpy().tobytes() for d in draws})
    check("workers and epochs draw different distortions", distinct == len(draws),
          f"{distinct}/{len(draws)} distinct")

    print("--- Channel augmentation (--channel-aug, Finding 16) ---")
    # Training-only (channel_aug.py). A three-file stand-in for MUSAN and the
    # RIRs, laid out as hpc/get_channel_aug.slurm leaves them. The output must
    # be what preprocess_waveform expects — mono, 16 kHz, at most MAX_LEN —
    # and every stage must be reachable: each codec, reverb and noise.
    import tempfile
    import soundfile as sf
    from channel_aug import CODECS, ChannelAug, codec, reverb
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for sub, name, n in (("musan/noise/free-sound", "noise-free-sound-0000.wav", 3 * SAMPLE_RATE),
                             ("musan/music/fma", "music-fma-0000.wav", 8 * SAMPLE_RATE),
                             ("musan/music/fma", "music-fma-0001.wav", 8 * SAMPLE_RATE),
                             ("musan/music/fma", "music-fma-0002.wav", 8 * SAMPLE_RATE),
                             ("RIRS_NOISES/simulated_rirs/smallroom/Room001", "r.wav", 4000),
                             ("RIRS_NOISES/real_rirs_isotropic_noises", "air_rir.wav", 4000),
                             ("RIRS_NOISES/real_rirs_isotropic_noises", "noise.wav", 4000)):
            (Path(tmp) / sub).mkdir(parents=True, exist_ok=True)
            sf.write(str(Path(tmp) / sub / name), 0.1 * rng.standard_normal(n), SAMPLE_RATE)
        # Licences as MUSAN writes them: one file CC BY, one CC BY-NC-SA, one
        # with no entry at all. Only the first may be used. free-sound's
        # LICENSE names no files: one statement for the whole directory.
        (Path(tmp) / "musan/music/fma/LICENSE").write_text(
            "music-fma-0000\n\"A\" (by B)\nCC BY 4.0\n" + "=" * 20 + "\n"
            "music-fma-0001\n\"C\" (by D)\nCC BY-NC-SA 3.0\n")
        (Path(tmp) / "musan/noise/free-sound/LICENSE").write_text(
            "All selected recordings were marked as in the Public Domain\n")
        aug = ChannelAug(tmp, before=RawBoost(5))
        used = sorted(Path(p).name for p, _ in aug.noises)
        check("uses only commercially licensed noise and music",
              used == ["music-fma-0000.wav", "noise-free-sound-0000.wav"], used)
        check("uses only the simulated RIRs", len(aug.rirs) == 1, len(aug.rirs))
        np.random.seed(0)
        outs = [aug(waveform, sample_rate) for _ in range(20)]
        ok = all(o.shape[0] == 1 and o.shape[1] <= MAX_LEN and bool(torch.isfinite(o).all())
                 and o.abs().max() <= 1 for o in outs)
        check("output is mono, <= MAX_LEN, finite, within [-1, 1]", ok, tuple(outs[0].shape))
        x = outs[0][0].numpy().astype(np.float64)
        for kind in CODECS:
            y = codec(x, kind, 0.5)
            check(f"codec {kind} keeps length and changes the clip",
                  len(y) == len(x) and not np.allclose(y, x), len(y))
        check("reverb keeps length", len(reverb(x, rng.standard_normal(4000))) == len(x))
        spec = preprocess_waveform(outs[0], SAMPLE_RATE, build_transform("raw"))
        check("preprocess_waveform accepts its output", tuple(spec.shape) == (1, MAX_LEN),
              tuple(spec.shape))

    print("--- Input degradation (Finding 14) ---")
    # Scoring-time only. Each must keep the length, stay finite, change the
    # clip, and be deterministic: one upload has to give one reading.
    import torchaudio.transforms as T
    clip16 = waveform[:1] if sample_rate == SAMPLE_RATE else \
        T.Resample(sample_rate, SAMPLE_RATE)(waveform[:1])
    for name in DEGRADATIONS:
        a, b = degrade_waveform(clip16, name), degrade_waveform(clip16, name)
        changed = name == "none" or not torch.equal(a, clip16)
        ok = (tuple(a.shape) == tuple(clip16.shape) and bool(torch.isfinite(a).all())
              and torch.equal(a, b) and changed)
        check(f"{name}: keeps shape, finite, deterministic"
              + ("" if name == "none" else ", changes the clip"), ok, tuple(a.shape))
    check("'none' is the identity", torch.equal(degrade_waveform(clip16, "none"), clip16))
    short = degrade_waveform(clip16[:, :1000], "opus")
    check("opus keeps a short clip's length", short.shape[1] == 1000, tuple(short.shape))

    print("--- ASVspoof 5 adapter (--extra-train asvspoof5) ---")
    # A two-clip corpus laid out as hpc/get_asvspoof5.slurm leaves it, with a
    # protocol in the README's ten-column format. The adapter has to hand the
    # shared dataset class the same labels and the same tensors LA does — the
    # clips here ARE the LA samples, so any difference is the adapter's fault.
    import shutil
    import tempfile

    import pandas as pd

    import asvspoof5
    from train_dp_avspoof import AVSpoofDataset, compute_class_weights
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "flac_T").mkdir()
        shutil.copy(SCRIPT_DIR / "LA_T_1000406.flac", root / "flac_T" / "T_0000000001.flac")
        shutil.copy(SAMPLE, root / "flac_T" / "T_0000000002.flac")
        (root / "ASVspoof5.train.tsv").write_text(
            "T_4850 T_0000000001 F - - - - bonafide bonafide -\n"
            "T_4851 T_0000000002 M - - - AC3 A05 spoof -\n")
        proto, audio_dir = asvspoof5.load_protocol("train", root=root)
        check("protocol maps keys and attacks",
              list(proto["label"]) == ["bonafide", "spoof"]
              and list(proto["system_id"]) == ["-", "A05"], list(proto["system_id"]))
        ds5 = AVSpoofDataset(None, audio_dir, build_transform(DEFAULT_FRONTEND), protocol=proto)
        x5, y5 = ds5[1]
        check("spoof clip yields the LA tensor and label 1",
              torch.equal(x5, specs[DEFAULT_FRONTEND]) and int(y5) == 1)
        concat = torch.utils.data.ConcatDataset([ds5, ds5])
        concat.protocol = pd.concat([ds5.protocol, ds5.protocol], ignore_index=True)
        w = compute_class_weights(concat, torch.device("cpu"))
        check("class weights computed over the concatenated corpus",
              len(concat) == 4 and torch.allclose(w, torch.tensor([1.0, 1.0])), w.tolist())

    print("--- Common Voice adapter (--extra-bonafide commonvoice) ---")
    # Laid out as SpeechFake ships it: one clip in each split plus one in
    # another language. Training must see only English train, calibration only
    # English test — a leak between the two is the failure this guards.
    import commonvoice
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for lang, split in [("en", "train"), ("en", "test"), ("fr", "train")]:
            rel = f"Real/CommonVoice/{lang}/{split}/{lang}_{split}_0/common_voice_{lang}_1.wav"
            (root / rel).parent.mkdir(parents=True)
            shutil.copy(SAMPLE, root / rel)
            rows.append(f"{rel},bonafide,-,-,-,{lang}")
        (root / "metadata" / "Real").mkdir(parents=True)
        (root / "metadata" / "Real" / "CommonVoice.csv").write_text(
            "file,label,generator,model,speaker,language\n" + "\n".join(rows) + "\n")
        proto, audio_dir = commonvoice.load_protocol("train", root=root)
        check("training sees English train only, labelled bona fide",
              len(proto) == 1 and "/en/train/" in proto["audio_file_name"][0]
              and list(proto["label"]) == ["bonafide"], list(proto["audio_file_name"]))
        held_out, _ = commonvoice.load_clips("en", "test", root=root)
        check("calibration split shares no clip with training",
              not set(held_out["file"]) & set(proto["audio_file_name"]))
        dscv = AVSpoofDataset(None, audio_dir, build_transform(DEFAULT_FRONTEND),
                              protocol=proto, suffix="")
        xcv, ycv = dscv[0]
        check("Common Voice clip yields the sample's tensor and label 0",
              torch.equal(xcv, specs[DEFAULT_FRONTEND]) and int(ycv) == 0)

    print("--- VoxPopuli adapter (--extra-bonafide commonvoice+voxpopuli) ---")
    # Speaker 1 is in train and test, as VoxPopuli's own splits allow; speaker
    # 2 is in test only. Train shard 00001 is outside the Finding 9 set.
    import voxpopuli
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [("train", "train-00000-of-00030", "1"), ("train", "train-00001-of-00030", "3"),
                ("test", "test-00000-of-00001", "1"), ("test", "test-00000-of-00001", "2")]
        lines = []
        for i, (split, shard, spk) in enumerate(rows):
            rel = f"audio/{split}/clip{i}.wav"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SAMPLE, root / rel)
            lines.append(f"{rel},{spk},male,{split},{shard}")
        (root / "clips.csv").write_text("file,speaker_id,gender,split,shard\n"
                                        + "\n".join(lines) + "\n")
        proto, _ = voxpopuli.load_protocol("train", root=root)
        check("training sees every train shard and nothing else, labelled bona fide",
              list(proto["audio_file_name"]) == ["audio/train/clip0.wav", "audio/train/clip1.wav"]
              and set(proto["label"]) == {"bonafide"}, list(proto["audio_file_name"]))
        held, _ = voxpopuli.load_clips("heldout", root=root)
        check("held-out split drops test speakers who are also in train",
              list(held["speaker_id"]) == ["2"], list(held["speaker_id"]))
        calib, _ = voxpopuli.load_clips("calibration", root=root)
        check("the Finding 9 calibration set ignores the shards added later",
              "audio/train/clip1.wav" not in set(calib["file"]) and len(calib) == 3)

    print("--- People's Speech adapter (calibration, Finding 11) ---")
    import peoples_speech
    from calibrate import trained_sources
    check("recording names matching an In-the-Wild speaker are caught",
          peoples_speech.itw_match("BarackObama_Address_DOT_flac") == "obama"
          and peoples_speech.itw_match("jfk_inaugural") == "jfk")
    check("ordinary words containing a surname are not",
          peoples_speech.itw_match("Trumpet_Lessons_Bushwick") == "")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "clips.csv").write_text(
            "file,speaker_id,duration_ms,itw_match\n"
            "audio/a.flac,Passive_Houses,1000,\n"
            "audio/b.flac,Obama_Speech,1000,obama\n")
        ps, _ = peoples_speech.load_clips(root=root)
        check("load_clips drops the In-the-Wild matches",
              list(ps["file"]) == ["audio/a.flac"], list(ps["file"]))
    check("calibrate.py reads every trained source off extra_bonafide",
          trained_sources({"extra_bonafide": "commonvoice+voxpopuli"})
          == {"commonvoice", "voxpopuli"} and trained_sources({}) == set())

    spec = specs[DEFAULT_FRONTEND]

    print("--- Train/serve parity ---")
    # app.py must produce a byte-identical tensor to the training dataset path.
    # It loads weights at import time, so if there is no usable checkpoint yet we
    # drop in a temporary one built from an untrained net — the parity check cares
    # about preprocessing, not about what the weights contain.
    # Same $CKPT_ROOT app.py and the trainer read, so this checks the tree that
    # will actually be served rather than an empty one next to the script.
    ckpt_dir = Path(os.getenv("CKPT_ROOT", SCRIPT_DIR / "checkpoints"))
    best = ckpt_dir / "best.pth"
    created_ckpt = not best.exists()
    if created_ckpt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # The metrics carry a numpy scalar on purpose. A real DP checkpoint
        # always does — Opacus's get_epsilon() returns one — and torch 2.6
        # changed torch.load's default to weights_only=True, which refuses any
        # checkpoint containing one. That broke app.py at import time on M3
        # while passing here, because a temporary checkpoint of plain tensors
        # is exactly the case the new default still allows. A stand-in that is
        # easier to load than the real thing tests nothing.
        torch.save({"epoch": 0, "steps_done": 0, "arch": DEFAULT_ARCH,
                    "frontend": DEFAULT_FRONTEND,
                    "model": build_model(DEFAULT_ARCH).state_dict(),
                    "metrics": {"epsilon": np.float64(0.48)}}, best)
        print("  (no trained weights found; using a temporary untrained checkpoint)")

    try:
        import app

        served = app.preprocess_audio(str(SAMPLE))
        served_frontend = app.FRONTEND
        check("app.py preprocessing matches training exactly",
              torch.allclose(served, specs[served_frontend].unsqueeze(0), atol=0))
        check(f"app.py output tensor shape matches the {served_frontend} front-end",
              tuple(served.shape) == (1,) + expected_shape[served_frontend],
              tuple(served.shape))
    finally:
        if created_ckpt:
            best.unlink()

    print()
    if failures:
        print(f"❌ {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("✅ All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
