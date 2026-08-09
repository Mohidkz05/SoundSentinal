# verify_setup.py
#
# Smoke test for the model/serving contract. Run this after changing anything in
# model.py, and after a fresh clone, before assuming the pipeline works:
#
#     python verify_setup.py
#
# It needs torch/torchaudio but does NOT need the ASVspoof dataset or trained
# weights — it uses the two sample .flac files committed alongside it.

import sys
from pathlib import Path

import torch
import torchaudio

from model import AudioClassifierCNN, build_transform, preprocess_waveform

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE = SCRIPT_DIR / "LA_T_1000137.flac"

failures = []


def check(name, condition, extra=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {extra}")
    if not condition:
        failures.append(name)


def main():
    print("--- Preprocessing ---")
    transform_pipeline = build_transform()
    waveform, sample_rate = torchaudio.load(str(SAMPLE))
    spec = preprocess_waveform(waveform, sample_rate, transform_pipeline)

    check("spectrogram shape is (1, 128, 126)", tuple(spec.shape) == (1, 128, 126), tuple(spec.shape))
    check("standardized to mean ~0", abs(float(spec.mean())) < 1e-4, f"mean={float(spec.mean()):.2e}")
    check("standardized to std ~1", abs(float(spec.std()) - 1.0) < 1e-2, f"std={float(spec.std()):.4f}")

    print("--- Model ---")
    net = AudioClassifierCNN().eval()
    check("fc1 is Linear(2048, 128)", tuple(net.fc1.weight.shape) == (128, 2048), tuple(net.fc1.weight.shape))

    with torch.no_grad():
        out = net(spec.unsqueeze(0))
    check("forward pass returns 2 logits", tuple(out.shape) == (1, 2), tuple(out.shape))

    # The whole point of AdaptiveAvgPool2d: input length must not matter.
    for frames in (63, 126, 400):
        with torch.no_grad():
            o = net(torch.randn(1, 1, 128, frames))
        check(f"handles spectrogram width {frames}", tuple(o.shape) == (1, 2), tuple(o.shape))

    print("--- Train/serve parity ---")
    # app.py must produce a byte-identical tensor to the training dataset path.
    try:
        import app
    except FileNotFoundError as e:
        print(f"  [SKIP] app.py load (no trained weights yet)\n         {e}")
    else:
        served = app.preprocess_audio(str(SAMPLE))
        check("app.py preprocessing matches training exactly",
              torch.allclose(served, spec.unsqueeze(0), atol=0))

    print()
    if failures:
        print(f"❌ {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("✅ All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
