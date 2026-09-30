# calibrate_clean.py
#
# Add a clean-audio route to an already-calibrated checkpoint (RESULTS.md
# Finding 15).
#
#     python calibrate_clean.py --ckpt <dir>/best_calibrated_peoples_speech_1pct.pth
#
# WHY. The served threshold was fitted on noisy real speech (People's
# Speech), and clean audio sits ~14 log-odds lower on the model's scale
# (Finding 13), so a third of clean fakes pass. Finding 14 showed degrading
# the input cannot move clean audio up. This moves the line instead, for
# clean recordings only: a second threshold, fitted the same way on clean real
# speech the model never trained on (LibriSpeech test-clean, librispeech.py),
# applied to clips whose model.cleanliness_db is at or above a cutoff.
#
# WHAT IT FITS, using only the FIT halves of both calibration sets:
#
#   1. the cutoff: the cleanliness value that best separates LibriSpeech
#      (clean) from People's Speech (not), by balanced accuracy;
#   2. the clean threshold: the score that --target-frr of the LibriSpeech
#      clips routed clean exceed — the same rule calibrate.py uses.
#
# The CHECK halves (whole speakers / recordings the fit never saw) then report
# the flag rates under routing. The noisy threshold is the checkpoint's own and
# is not changed. Output: a copy of the checkpoint with a `routing` block,
# which evaluate.py and app.py apply, plus a JSON report beside it.
#
# WHAT IT MUST NOT DO. Read In-the-Wild, SpeechFake or LA. The cutoff and the
# clean threshold come from real speech alone, as calibrate.py's threshold
# does; how many fakes pass is measured afterwards by evaluate.py.

import argparse
import json
import os
from pathlib import Path
from time import strftime

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate import LOADERS, flagged, split_halves, to_protocol
from evaluate import BATCH_SIZE, DEVICE, log_odds_to_prob, prob_to_log_odds, score_dataset
from model import DEFAULT_ARCH, DEFAULT_FRONTEND, build_model, build_transform
import librispeech
from train_dp_avspoof import AVSpoofDataset


def score(model, frame, root, frontend, num_workers, degradation):
    dataset = AVSpoofDataset(None, root, build_transform(frontend), protocol=frame, suffix="",
                             degradation=degradation, with_cleanliness=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    scores, _, clean = score_dataset(model, loader, with_cleanliness=True)
    return scores, clean


def best_cutoff(clean_db, noisy_db):
    """The cutoff c maximising balanced accuracy of `clean if db >= c`."""
    candidates = np.unique(np.concatenate([clean_db, noisy_db]))
    best, best_acc = None, -1.0
    for c in candidates:
        acc = 0.5 * ((clean_db >= c).mean() + (noisy_db < c).mean())
        if acc > best_acc:
            best, best_acc = float(c), float(acc)
    return best, best_acc


def pct(x):
    return {str(q): float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95, 99)} if len(x) else {}


def main():
    parser = argparse.ArgumentParser(description="Add a clean-audio threshold route.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="A checkpoint already calibrated by calibrate.py "
                             "(its threshold becomes the noisy route's).")
    parser.add_argument("--target-frr", type=float, default=0.01)
    parser.add_argument("--n", type=int, default=10000,
                        help="People's Speech clips, as in calibrate.py (same sample).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int,
                        default=int(os.getenv("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: <ckpt dir>/<stem>_routed.pth")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    if ckpt.get("calibration", {}).get("calibration_set", "").split(" ")[0] != "peoples_speech":
        raise SystemExit("ERROR: --ckpt must be calibrated on People's Speech first "
                         "(calibrate.py --calibration-set peoples_speech).")
    if "librispeech" in (ckpt.get("extra_bonafide") or ""):
        raise SystemExit("ERROR: this checkpoint trained on LibriSpeech; it is not held out.")
    noisy_thr = prob_to_log_odds(ckpt["threshold"])
    degradation = ckpt.get("input_degradation") or "none"

    arch = ckpt.get("arch") or DEFAULT_ARCH
    frontend = ckpt.get("frontend") or DEFAULT_FRONTEND
    model = build_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    ps_clips, ps_root = LOADERS["peoples_speech"]("en")
    ps_fit, ps_check, _ = split_halves(ps_clips, args.n, args.seed)
    ls_clips, ls_root = librispeech.load_clips()
    ls_fit, ls_check, _ = split_halves(ls_clips, len(ls_clips), args.seed)

    print(f"Checkpoint   {args.ckpt} (epoch {ckpt.get('epoch')})")
    print(f"Noisy route  log-odds {noisy_thr:+.2f} (the checkpoint's own threshold)")
    print(f"People's Speech {len(ps_fit)} fit / {len(ps_check)} check; "
          f"LibriSpeech {len(ls_fit)} fit / {len(ls_check)} check (seed {args.seed})\n")

    s_ps_fit, c_ps_fit = score(model, to_protocol(ps_fit), ps_root, frontend, args.num_workers, degradation)
    s_ps_chk, c_ps_chk = score(model, to_protocol(ps_check), ps_root, frontend, args.num_workers, degradation)
    s_ls_fit, c_ls_fit = score(model, to_protocol(ls_fit), ls_root, frontend, args.num_workers, degradation)
    s_ls_chk, c_ls_chk = score(model, to_protocol(ls_check), ls_root, frontend, args.num_workers, degradation)

    # 1. The cutoff, from the fit halves only.
    cutoff, bal_acc = best_cutoff(c_ls_fit, c_ps_fit)

    # 2. The clean threshold: calibrate.py's rule, on LibriSpeech fit clips
    #    that the cutoff routes clean.
    routed = s_ls_fit[c_ls_fit >= cutoff]
    clean_thr = float(np.quantile(routed, 1 - args.target_frr, method="higher"))

    def routed_flag(scores, clean):
        thr = np.where(clean >= cutoff, clean_thr, noisy_thr)
        return float((scores >= thr).mean())

    ls_chk_routed = s_ls_chk[c_ls_chk >= cutoff]
    report = {
        "timestamp": strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(args.ckpt), "epoch": ckpt.get("epoch"),
        "seed": args.seed, "target_frr": args.target_frr,
        "input_degradation": degradation,
        "cutoff_db": cutoff, "cutoff_balanced_accuracy_fit": bal_acc,
        "noisy_threshold_log_odds": noisy_thr,
        "clean_threshold_log_odds": clean_thr,
        "clean_threshold": log_odds_to_prob(clean_thr),
        "routed_clean": {
            "librispeech_fit": float((c_ls_fit >= cutoff).mean()),
            "librispeech_check": float((c_ls_chk >= cutoff).mean()),
            "peoples_speech_fit": float((c_ps_fit >= cutoff).mean()),
            "peoples_speech_check": float((c_ps_chk >= cutoff).mean()),
        },
        "flagged_under_routing": {
            "librispeech_fit": routed_flag(s_ls_fit, c_ls_fit),
            "librispeech_check": routed_flag(s_ls_chk, c_ls_chk),
            "peoples_speech_fit": routed_flag(s_ps_fit, c_ps_fit),
            "peoples_speech_check": routed_flag(s_ps_chk, c_ps_chk),
        },
        "flagged_single_threshold": {
            "librispeech_check": flagged(s_ls_chk, noisy_thr),
            "peoples_speech_check": flagged(s_ps_chk, noisy_thr),
        },
        "cleanliness_db_percentiles": {
            "librispeech": pct(np.concatenate([c_ls_fit, c_ls_chk])),
            "peoples_speech": pct(np.concatenate([c_ps_fit, c_ps_chk])),
        },
        "clean_check_score_percentiles_log_odds": pct(ls_chk_routed),
    }

    r = report
    print(f"Cutoff       cleanliness >= {cutoff:.1f} dB routes clean "
          f"(balanced accuracy {bal_acc:.1%} on the fit halves)")
    print(f"  routed clean: LibriSpeech {r['routed_clean']['librispeech_check']:.1%}, "
          f"People's Speech {r['routed_clean']['peoples_speech_check']:.1%} (check halves)")
    print(f"Clean route  log-odds {clean_thr:+.2f} (P = {log_odds_to_prob(clean_thr):.6g}); "
          f"noisy route {noisy_thr:+.2f}")
    print("Real speech flagged, check halves      routed    single threshold")
    for name, key in (("LibriSpeech", "librispeech_check"), ("People's Speech", "peoples_speech_check")):
        print(f"  {name:<36} {r['flagged_under_routing'][key]:6.2%}    "
              f"{r['flagged_single_threshold'][key]:6.2%}")

    ckpt["routing"] = {
        "cutoff_db": cutoff,
        "threshold_clean": log_odds_to_prob(clean_thr),
        "threshold_clean_log_odds": clean_thr,
        # The clean route's uncertain band starts where held-out clean real
        # speech stops being typical, as the noisy route's does (app.py).
        "clean_band_low": (float(np.percentile(ls_chk_routed, 95))
                           if len(ls_chk_routed) else None),
        "source": (f"{args.target_frr:.0%} of genuine LibriSpeech test-clean clips flagged, "
                   f"for recordings at or above {cutoff:.0f} dB dynamic range"),
        "report": report,
    }
    out = args.out or args.ckpt.with_name(f"{args.ckpt.stem}_routed.pth")
    torch.save(ckpt, out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}\nWrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
