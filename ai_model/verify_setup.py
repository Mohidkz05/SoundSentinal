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

from model import (ARCH_FRONTENDS, ARCHITECTURES, DEFAULT_ARCH,
                   DEFAULT_FRONTEND, FRONTENDS, MAX_LEN, N_LFCC, N_MELS,
                   build_model, build_transform, check_pairing,
                   default_frontend_for, load_audio, preprocess_waveform)

from rawboost import ALGOS as RAWBOOST_ALGOS, RawBoost

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE = SCRIPT_DIR / "LA_T_1000137.flac"

# Published parameter counts, from Table 2 of the AASIST paper and the official
# repo. Ours differ by exactly the 512 dead bn1 weights that deviation 2 in
# aasist.py removes, so these are the numbers to expect rather than 297k/85k.
EXPECTED_PARAMS = {"cnn": 267_330, "aasist": 297_354, "aasist-l": 85_034}

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

        for arch in ARCHITECTURES:
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
