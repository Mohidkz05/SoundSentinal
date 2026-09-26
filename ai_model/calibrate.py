# calibrate.py
#
# Re-derive a checkpoint's decision threshold from real-world bona fide speech.
#
#     python calibrate.py --ckpt <dir>/best.pth                  # 5% of real clips flagged
#     python calibrate.py --ckpt <dir>/best.pth --target-frr 0.02
#
# WHY. The trainer calibrates the threshold at the dev set's equal error rate.
# For SSL-AASIST trained on LA + SpeechFake that threshold is P(spoof) = 0.0026,
# and it flags 46% of genuine In-the-Wild clips (RESULTS.md Finding 9). The
# model ranks In-the-Wild well (2.65% EER) — the SCALE is what fails to
# transfer: dev's bona fide is clean read speech (VCTK, LibriTTS, AISHELL), and
# real-world recordings score far higher on "spoof" than clean ones, while
# still scoring below real-world fakes.
#
# WHAT. Common Voice: volunteers reading sentences on their own microphones in
# their own rooms (CC0, shipped as SpeechFake's Real/CommonVoice.zip). No
# training or dev protocol references it, so the model has never seen it. The
# English subset is sampled, split in two, and scored:
#
#   - the CALIBRATION half sets the threshold, so that --target-frr of its
#     clips (default 5%) are flagged as fake;
#   - the CHECK half, never used to choose anything, confirms the rate holds
#     on clips the threshold was not fitted to.
#
# The threshold is set from bona fide clips alone. That is deliberate: it
# fixes the rate at which a real voice gets wrongly flagged, which is the
# error a user of this tool experiences as an accusation. The rate at which
# fakes slip through is then MEASURED, not chosen — by evaluate.py on
# In-the-Wild, at the threshold this script writes.
#
# WHAT IT MUST NOT DO. Touch In-the-Wild. Calibrating on the test set would
# make every In-the-Wild number meaningless. The target rate is fixed before
# In-the-Wild is scored at the new threshold, and is not revised after.
#
# Output: a COPY of the checkpoint with the new threshold (the original keeps
# its dev-EER one, so every earlier number stays reproducible), plus a JSON
# report beside it. app.py reads `threshold` and `threshold_source` from it.
#
# Nothing here redefines the network or the preprocessing: the dataset class
# and the scoring loop are imported, see "The one rule" in CLAUDE.md.

import argparse
import json
import os
from pathlib import Path
from time import strftime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate import BATCH_SIZE, DEVICE, log_odds_to_prob, prob_to_log_odds, score_dataset
from model import DEFAULT_ARCH, DEFAULT_FRONTEND, build_model, build_transform
from speechfake import get_speechfake_root
from train_dp_avspoof import AVSpoofDataset

COMMONVOICE_CSV = Path("metadata") / "Real" / "CommonVoice.csv"


def load_commonvoice(root, language, n, seed):
    """n bona fide Common Voice clips in `language`, as two halves of a five-column frame."""
    meta = pd.read_csv(root / COMMONVOICE_CSV, dtype=str)
    meta = meta[(meta["language"] == language) & (meta["label"] == "bonafide")]
    if len(meta) < n:
        raise ValueError(f"Only {len(meta)} {language} clips in {COMMONVOICE_CSV}, asked for {n}.")
    missing = [f for f in meta["file"].head(20) if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root / missing[0]} does not exist. Was Real/CommonVoice.zip extracted?\n"
            f"Fetch it:  sbatch --export=ALL,WITH_COMMONVOICE=1 hpc/get_speechfake.slurm")
    sample = meta.sample(n=n, random_state=seed).reset_index(drop=True)
    frame = pd.DataFrame({
        "speaker_id": "-",           # Common Voice ships no speaker ids here
        "audio_file_name": sample["file"],
        "_": "-",
        "system_id": "-",
        "label": "bonafide",
    })
    half = n // 2
    return frame.iloc[:half].reset_index(drop=True), frame.iloc[half:].reset_index(drop=True)


def score(model, frame, root, frontend, num_workers):
    dataset = AVSpoofDataset(None, root, build_transform(frontend), protocol=frame, suffix="")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    scores, _ = score_dataset(model, loader)
    return scores


def flagged(scores, threshold_log_odds):
    """Fraction of bona fide clips called spoof — `spoof if score >= threshold`, as app.py does."""
    return float((scores >= threshold_log_odds).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Set a checkpoint's threshold from real-world bona fide speech (Common Voice).")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--target-frr", type=float, default=0.05,
                        help="Fraction of genuine clips the threshold may flag as fake. "
                             "A product decision, fixed before In-the-Wild is scored.")
    parser.add_argument("--language", default="en",
                        help="Common Voice language to calibrate on. English matches "
                             "In-the-Wild and the app's expected input.")
    parser.add_argument("--n", type=int, default=10000,
                        help="Clips to sample; half calibrate, half check.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int,
                        default=int(os.getenv("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--out", type=Path, default=None,
                        help="Calibrated checkpoint. Default: <ckpt dir>/<stem>_calibrated.pth")
    args = parser.parse_args()
    if not 0 < args.target_frr < 1:
        raise SystemExit("--target-frr must be between 0 and 1")

    root = get_speechfake_root()
    calib, check = load_commonvoice(root, args.language, args.n, args.seed)

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch") or DEFAULT_ARCH
    frontend = ckpt.get("frontend") or DEFAULT_FRONTEND
    model = build_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    dev_threshold = ckpt.get("threshold")

    print(f"Checkpoint   {args.ckpt} (epoch {ckpt.get('epoch')}, {arch})")
    print(f"Calibration  Common Voice {args.language}: {len(calib)} clips to fit, "
          f"{len(check)} to check (seed {args.seed})")
    print(f"Target       flag {args.target_frr:.1%} of genuine clips\n")

    calib_scores = score(model, calib, root, frontend, args.num_workers)
    check_scores = score(model, check, root, frontend, args.num_workers)

    # The (1 - target) quantile of the bona fide scores: everything at or above
    # it is flagged. "higher" picks an observed score, so the fitted rate is
    # at most the target rather than just over it.
    thr = float(np.quantile(calib_scores, 1 - args.target_frr, method="higher"))
    thr_prob = log_odds_to_prob(thr)

    report = {
        "timestamp": strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(args.ckpt),
        "epoch": ckpt.get("epoch"),
        "calibration_set": f"Common Voice ({args.language}), via SpeechFake Real/CommonVoice.zip",
        "n_calibrate": len(calib), "n_check": len(check), "seed": args.seed,
        "target_frr": args.target_frr,
        "threshold": thr_prob, "threshold_log_odds": thr,
        "frr_calibrate": flagged(calib_scores, thr),
        "frr_check": flagged(check_scores, thr),
        "dev_threshold": dev_threshold,
        "frr_check_at_dev_threshold": (flagged(check_scores, prob_to_log_odds(dev_threshold))
                                       if dev_threshold is not None else None),
        "check_score_percentiles_log_odds": {
            str(q): float(np.percentile(check_scores, q)) for q in (5, 25, 50, 75, 95, 99)},
    }

    print(f"New threshold          P(spoof) = {thr_prob:.6g}  (log-odds {thr:+.2f})")
    print(f"  flagged, fit half    {report['frr_calibrate']:6.2%}")
    print(f"  flagged, check half  {report['frr_check']:6.2%}   <- the number to trust")
    if dev_threshold is not None:
        print(f"Old dev-EER threshold  P(spoof) = {dev_threshold:.4g} flags "
              f"{report['frr_check_at_dev_threshold']:6.2%} of the check half")
    print("\nCheck-half scores (log-odds), percentiles 5/25/50/75/95/99:")
    print("  " + "  ".join(f"{v:+.2f}" for v in report["check_score_percentiles_log_odds"].values()))

    out = args.out or args.ckpt.with_name(f"{args.ckpt.stem}_calibrated.pth")
    ckpt["threshold_dev"] = dev_threshold
    ckpt["threshold"] = thr_prob
    ckpt["threshold_source"] = (f"{args.target_frr:.0%} of genuine Common Voice "
                                f"({args.language}) clips flagged")
    ckpt["calibration"] = report
    torch.save(ckpt, out)
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}\nWrote {report_path}")
    print("Next: score In-the-Wild ONCE at this threshold with evaluate.py --ckpt <that file>.")


if __name__ == "__main__":
    main()
