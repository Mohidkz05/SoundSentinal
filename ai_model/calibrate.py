# calibrate.py
#
# Re-derive a checkpoint's decision threshold from real-world bona fide speech.
#
#     python calibrate.py --ckpt <dir>/best.pth                  # Common Voice, 5% flagged
#     python calibrate.py --ckpt <dir>/best.pth --calibration-set voxpopuli
#     python calibrate.py --ckpt <dir>/best.pth --target-frr 0.02
#     python calibrate.py --ckpt <dir>/best.pth --commonvoice-split test   # see below
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
# English subset is sampled, split in two, and scored. `--calibration-set
# voxpopuli` uses European Parliament speeches instead (voxpopuli.py): after
# Common Voice left In-the-Wild's false flags at 15.6%, the closer match to
# In-the-Wild's speeches and broadcast audio. Where the set has speaker ids,
# the two halves share no speaker, so the check half is voices the threshold
# never saw.
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
# A MODEL TRAINED ON COMMON VOICE (--extra-bonafide commonvoice) saw the
# English train split labelled bona fide, so calibrating on those clips would
# fit the threshold to data the model was taught to call real. For such a
# checkpoint this script requires --commonvoice-split test — Common Voice's own
# held-out split, whose speakers are not in train (commonvoice.py). The earlier
# calibrations sampled both splits (--commonvoice-split all, the default), and
# are reproduced unchanged.
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
import commonvoice
import voxpopuli
from train_dp_avspoof import AVSpoofDataset

def load_commonvoice(language, split="all"):
    """Bona fide Common Voice clips in `language` (commonvoice.py). No speaker ids."""
    clips, root = commonvoice.load_clips(language, None if split == "all" else split)
    return clips[["file", "speaker_id"]], root


def load_voxpopuli(language, split="all"):
    if language != "en":
        raise ValueError("Only VoxPopuli English is fetched (hpc/get_voxpopuli.slurm).")
    clips, root = voxpopuli.load_clips()
    return clips[["file", "speaker_id"]], root


LOADERS = {"commonvoice": load_commonvoice, "voxpopuli": load_voxpopuli}


def split_halves(clips, n, seed):
    """Sample n clips and split them into (calibrate, check) halves.

    With speaker ids, whole speakers are assigned to one half — so the check
    half measures the rate on voices the threshold was not fitted to, not on
    new sentences from the same voices. Without them (Common Voice here), a
    random split.
    """
    if len(clips) < n:
        raise ValueError(f"Only {len(clips)} clips available, asked for {n}.")
    sample = clips.sample(n=n, random_state=seed).reset_index(drop=True)
    if (sample["speaker_id"] == "-").all():
        return sample.iloc[: n // 2], sample.iloc[n // 2:], "random"
    speakers = sample["speaker_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
    sizes = sample["speaker_id"].value_counts()
    calib_speakers, total = set(), 0
    for spk in speakers:
        if total >= n // 2:
            break
        calib_speakers.add(spk)
        total += sizes[spk]
    in_calib = sample["speaker_id"].isin(calib_speakers)
    return sample[in_calib], sample[~in_calib], "by speaker"


def to_protocol(clips):
    """The five-column frame AVSpoofDataset reads."""
    return pd.DataFrame({
        "speaker_id": clips["speaker_id"].values,
        "audio_file_name": clips["file"].values,
        "_": "-",
        "system_id": "-",
        "label": "bonafide",
    })


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
        description="Set a checkpoint's threshold from real-world bona fide speech.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--target-frr", type=float, default=0.05,
                        help="Fraction of genuine clips the threshold may flag as fake. "
                             "A product decision, fixed before In-the-Wild is scored.")
    parser.add_argument("--calibration-set", default="commonvoice", choices=sorted(LOADERS),
                        help="Real speech to calibrate on. commonvoice: people reading at "
                             "home. voxpopuli: European Parliament speeches, closer to "
                             "In-the-Wild's speeches and broadcast audio.")
    parser.add_argument("--language", default="en",
                        help="Language to calibrate on. English matches In-the-Wild "
                             "and the app's expected input.")
    parser.add_argument("--commonvoice-split", default="all", choices=["all", *commonvoice.SPLITS],
                        help="Which Common Voice split to sample from. 'all' reproduces "
                             "the calibrations in RESULTS.md Finding 9; a checkpoint "
                             "trained with --extra-bonafide commonvoice must use 'test'.")
    parser.add_argument("--n", type=int, default=10000,
                        help="Clips to sample; half calibrate, half check.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int,
                        default=int(os.getenv("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--out", type=Path, default=None,
                        help="Calibrated checkpoint. Default: "
                             "<ckpt dir>/<stem>_calibrated_<calibration set>.pth")
    args = parser.parse_args()
    if not 0 < args.target_frr < 1:
        raise SystemExit("--target-frr must be between 0 and 1")

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    # Before any scoring: calibrating on the clips a model was trained to call
    # real would make the "check half" rate meaningless.
    if (ckpt.get("extra_bonafide") == args.calibration_set == "commonvoice"
            and args.commonvoice_split != "test"):
        raise SystemExit(
            "ERROR: this checkpoint trained on Common Voice's train split. "
            "Calibrate on its held-out split: pass --commonvoice-split test.")
    if args.calibration_set != "commonvoice" and args.commonvoice_split != "all":
        raise SystemExit("--commonvoice-split only applies to --calibration-set commonvoice.")

    clips, root = LOADERS[args.calibration_set](args.language, args.commonvoice_split)
    calib_clips, check_clips, split_kind = split_halves(clips, args.n, args.seed)
    calib, check = to_protocol(calib_clips), to_protocol(check_clips)

    arch = ckpt.get("arch") or DEFAULT_ARCH
    frontend = ckpt.get("frontend") or DEFAULT_FRONTEND
    model = build_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    dev_threshold = ckpt.get("threshold")

    print(f"Checkpoint   {args.ckpt} (epoch {ckpt.get('epoch')}, {arch})")
    print(f"Calibration  {args.calibration_set} ({args.language}): {len(calib)} clips to fit, "
          f"{len(check)} to check, split {split_kind} (seed {args.seed})")
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
        "calibration_set": f"{args.calibration_set} ({args.language})",
        "commonvoice_split": (args.commonvoice_split
                              if args.calibration_set == "commonvoice" else None),
        "n_calibrate": len(calib), "n_check": len(check), "seed": args.seed,
        "split": split_kind,
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

    out = args.out or args.ckpt.with_name(f"{args.ckpt.stem}_calibrated_{args.calibration_set}.pth")
    ckpt["threshold_dev"] = dev_threshold
    ckpt["threshold"] = thr_prob
    names = {"commonvoice": "Common Voice", "voxpopuli": "VoxPopuli"}
    ckpt["threshold_source"] = (f"{args.target_frr:.0%} of genuine {names[args.calibration_set]} "
                                f"({args.language}) clips flagged")
    ckpt["calibration"] = report
    torch.save(ckpt, out)
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}\nWrote {report_path}")
    print("Next: score In-the-Wild ONCE at this threshold with evaluate.py --ckpt <that file>.")


if __name__ == "__main__":
    main()
