# evaluate.py
#
# Score a trained checkpoint against the ASVspoof2019 EVAL partition.
#
#     python evaluate.py                 # the --no-dp baseline (checkpoints/nodp)
#     python evaluate.py --dp            # the DP run (checkpoints/)
#     python evaluate.py --ckpt path.pth # any specific checkpoint
#
# This exists because dev EER is not a result. The dev partition is built from
# attacks A01–A06, the same six the model trains on, so a good dev number can
# mean the model recognised six specific vocoders rather than that it detects
# synthetic speech. Eval holds A07–A19, none of them seen in training, and every
# row of the comparison table in APPROACH.md is an eval number. Quoting dev
# beside them compares different quantities.
#
# Nothing here redefines the network or the preprocessing — both come from
# model.py, and the dataset and EER routine come from the trainer, so a change
# to preprocessing cannot make training and evaluation disagree. See "The one
# rule" in CLAUDE.md.

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from time import strftime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AudioClassifierCNN, CLASS_NAMES, LABEL_MAP, build_transform
from train_dp_avspoof import (
    AVSpoofDataset,
    CKPT_DIR,
    compute_eer_np,
    get_corpus_paths,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Larger than the training batch: no gradients are retained, so the only limit
# is activation memory for one forward pass.
BATCH_SIZE = 128


def confusion_at(labels, scores, threshold):
    """Counts at a given operating point. `spoof if p >= threshold`, matching app.py."""
    pred = (scores >= threshold).astype(int)
    return {
        "tn": int(((labels == 0) & (pred == 0)).sum()),
        "fp": int(((labels == 0) & (pred == 1)).sum()),
        "fn": int(((labels == 1) & (pred == 0)).sum()),
        "tp": int(((labels == 1) & (pred == 1)).sum()),
    }


def rates_at(labels, scores, threshold):
    """False-accept and false-reject rates at an operating point.

    Named the way the anti-spoofing literature does rather than the way sklearn
    does: a "false accept" is a spoof let through, which is the error that
    matters for a detector, and keeping the name makes the number comparable to
    what papers report.
    """
    cm = confusion_at(labels, scores, threshold)
    far = cm["fp"] / max(1, cm["fp"] + cm["tn"])   # bonafide wrongly flagged
    frr = cm["fn"] / max(1, cm["fn"] + cm["tp"])   # spoof wrongly passed
    return cm, far, frr


@torch.no_grad()
def score_dataset(model, loader):
    """Run the model over a loader, returning P(spoof) and the true labels."""
    model.eval()
    all_scores, all_labels = [], []
    total = len(loader.dataset)
    seen = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        probs = F.softmax(model(x), dim=1)
        all_scores.append(probs[:, 1].cpu())
        all_labels.append(y)
        seen += y.size(0)
        # A plain periodic line rather than a progress bar: this runs in a Slurm
        # job where stdout is a file, and a bar would write one line per update.
        if seen % (BATCH_SIZE * 50) == 0 or seen == total:
            print(f"  scored {seen}/{total}", flush=True)
    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def per_attack_eer(labels, scores, system_ids):
    """EER for each attack on its own, each against the full bonafide set.

    This is how ASVspoof papers break results down, and it is the part worth
    reading: a single pooled EER hides that a detector can be near-perfect on
    most attacks and blind to one. A03 at 40% and everything else at 2% is a
    completely different finding from a uniform 5%, and only this table
    distinguishes them.
    """
    bonafide = labels == 0
    out = OrderedDict()
    for attack in sorted(set(system_ids[labels == 1])):
        mask = bonafide | ((labels == 1) & (system_ids == attack))
        eer, thr = compute_eer_np(labels[mask], scores[mask])
        out[str(attack)] = {
            "eer": eer,
            "threshold": thr,
            "n_spoof": int(((labels == 1) & (system_ids == attack)).sum()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Score a checkpoint on the ASVspoof2019 eval partition.")
    parser.add_argument("--corpus", default="LA", choices=["LA", "PA"])
    parser.add_argument("--dp", action="store_true",
                        help="Score the DP run (checkpoints/) instead of the "
                             "--no-dp baseline (checkpoints/nodp/).")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="An explicit checkpoint path, overriding --dp.")
    parser.add_argument("--partition", default="eval", choices=["eval", "dev"],
                        help="Which partition to score. Defaults to eval, which "
                             "is the only one worth quoting; dev is offered to "
                             "reproduce a training run's own number.")
    parser.add_argument("--num-workers", type=int,
                        default=int(os.getenv("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the JSON result. Defaults to "
                             "<checkpoint dir>/eval_<partition>_<timestamp>.json")
    args = parser.parse_args()

    ckpt_path = args.ckpt or (CKPT_DIR if args.dp else CKPT_DIR / "nodp") / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            f"Train one first:  python train_dp_avspoof.py --corpus {args.corpus} --no-dp")

    if DEVICE.type == "cuda":
        print(f"Device: cuda -> {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    else:
        print(f"Device: CPU (no CUDA visible), torch {torch.__version__}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = AudioClassifierCNN().to(DEVICE)
    model.load_state_dict(ckpt["model"])

    # The threshold the checkpoint carries was calibrated on dev. It is the one
    # the server actually applies, so it is the honest deployment operating
    # point — at serving time there are no eval labels to tune against.
    dev_threshold = ckpt.get("threshold")
    calibrated = dev_threshold is not None
    if not calibrated:
        dev_threshold = 0.5

    print(f"Checkpoint    {ckpt_path}")
    print(f"  epoch       {ckpt.get('epoch')}")
    print(f"  regime      {'DP' if ckpt.get('dp') else 'non-private'}")
    print(f"  dev EER     {(ckpt.get('best_eer') or float('nan')) * 100:.2f}%")
    print(f"  threshold   {dev_threshold:.4f} "
          f"({'calibrated on dev' if calibrated else 'DEFAULT 0.5, checkpoint carries none'})")

    paths = get_corpus_paths(args.corpus)
    key = args.partition.upper()
    if f"{key}_AUDIO_DIR" not in paths:
        raise FileNotFoundError(
            f"No {args.partition} partition under $ASVSPOOF_ROOT for {args.corpus}. "
            f"Expected ASVspoof2019_{args.corpus}_{args.partition}/flac and a matching protocol.")

    dataset = AVSpoofDataset(
        paths[f"{key}_PROTOCOL_FILE"], paths[f"{key}_AUDIO_DIR"], build_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    counts = dataset.protocol["label"].map(LABEL_MAP).value_counts()
    print(f"\nPartition     {args.partition} ({len(dataset)} clips: "
          f"{int(counts.get(0, 0))} bonafide, {int(counts.get(1, 0))} spoof)")
    print(f"Attacks       {', '.join(sorted(set(dataset.protocol['system_id'])))}")
    print(f"\nScoring with {args.num_workers} workers...")

    scores, labels = score_dataset(model, loader)
    system_ids = dataset.protocol["system_id"].to_numpy()

    # Two operating points, and the distance between them is the point.
    pooled_eer, eer_threshold = compute_eer_np(labels, scores)
    cm_oracle, far_oracle, frr_oracle = rates_at(labels, scores, eer_threshold)
    cm_dev, far_dev, frr_dev = rates_at(labels, scores, dev_threshold)
    acc_dev = (cm_dev["tn"] + cm_dev["tp"]) / max(1, len(labels))

    print(f"\n=== {args.corpus} {args.partition} ===")
    print(f"EER                      {pooled_eer*100:6.2f}%  (at its own threshold {eer_threshold:.4f})")
    print(f"  confusion @EER         bonafide {cm_oracle['tn']} ok / {cm_oracle['fp']} flagged | "
          f"spoof {cm_oracle['tp']} caught / {cm_oracle['fn']} missed")
    print(f"\nAt the served threshold {dev_threshold:.4f} (calibrated on dev):")
    print(f"  accuracy               {acc_dev*100:6.2f}%")
    print(f"  false accept (spoof passed)   {frr_dev*100:6.2f}%")
    print(f"  false reject (real flagged)   {far_dev*100:6.2f}%")
    print(f"  confusion              bonafide {cm_dev['tn']} ok / {cm_dev['fp']} flagged | "
          f"spoof {cm_dev['tp']} caught / {cm_dev['fn']} missed")
    print("\n  The distance between these two blocks is the cost of calibrating on dev")
    print("  and deploying against unseen attacks. The EER line is what compares to")
    print("  published numbers; the threshold block is what a user would experience.")

    by_attack = per_attack_eer(labels, scores, system_ids)
    if by_attack:
        print(f"\n=== EER by attack (each against all bonafide) ===")
        worst = max(by_attack.items(), key=lambda kv: kv[1]["eer"])
        best = min(by_attack.items(), key=lambda kv: kv[1]["eer"])
        for attack, r in by_attack.items():
            mark = "  <- worst" if attack == worst[0] else ("  <- best" if attack == best[0] else "")
            print(f"  {attack:<6} {r['eer']*100:6.2f}%   n={r['n_spoof']:<6}{mark}")
        print(f"\n  Spread {best[1]['eer']*100:.2f}% – {worst[1]['eer']*100:.2f}%. A wide spread means"
              f"\n  the pooled EER above is an average over attacks the model handles very"
              f"\n  differently, which is worth saying explicitly in a writeup.")

    result = {
        "timestamp": strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt_path),
        "corpus": args.corpus,
        "partition": args.partition,
        "regime": "dp" if ckpt.get("dp") else "non-private",
        "epoch": ckpt.get("epoch"),
        "n_clips": int(len(labels)),
        "class_names": CLASS_NAMES,
        "dev": {"eer": ckpt.get("best_eer"), "threshold": dev_threshold,
                "threshold_calibrated": calibrated},
        "pooled": {
            "eer": pooled_eer, "eer_threshold": eer_threshold,
            "confusion_at_eer": cm_oracle,
            "at_served_threshold": {
                "threshold": dev_threshold, "accuracy": acc_dev,
                "false_accept_rate": frr_dev, "false_reject_rate": far_dev,
                "confusion": cm_dev,
            },
        },
        "by_attack": by_attack,
    }

    # Written to disk because metrics that only reach stdout vanish with the
    # Slurm log, and APPROACH.md's comparison table has to be assembled from
    # several runs. A JSON per run is the smallest thing that makes that
    # mechanical rather than a matter of scrolling back.
    out = args.out or ckpt_path.parent / f"eval_{args.partition}_{strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
