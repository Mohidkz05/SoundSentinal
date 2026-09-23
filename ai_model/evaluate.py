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

from model import (ARCHITECTURES, CLASS_NAMES, DEFAULT_ARCH, DEFAULT_FRONTEND,
                   FRONTENDS, LABEL_MAP, build_model, build_transform,
                   default_frontend_for)
from in_the_wild import get_itw_root, load_protocol as load_itw_protocol
from tdcf import compute_min_tdcf
from train_dp_avspoof import (
    AVSpoofDataset,
    CKPT_DIR,
    compute_eer_np,
    get_ckpt_paths,
    get_corpus_paths,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Larger than the training batch: no gradients are retained, so the only limit
# is activation memory for one forward pass.
BATCH_SIZE = 128


def fmt_pct(v):
    return "   n/a" if v is None else f"{v * 100:6.2f}%"


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
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=list(ARCHITECTURES),
                        help="Which architecture's run to score. Each one has its "
                             "own checkpoint directory, so this is how you reach "
                             "checkpoints/aasist/nodp/best.pth without naming it.")
    parser.add_argument("--frontend", default=None, choices=list(FRONTENDS),
                        help="Which front-end's run to score, defaulting to the "
                             "architecture's own. This is how you reach the LFCC "
                             "ablation at checkpoints/lfcc/nodp/best.pth.")
    parser.add_argument("--rawboost", type=int, default=None,
                        help="Score the run trained with this RawBoost algo — it lives "
                             "in its own checkpoint directory. Picks the checkpoint "
                             "only; no augmentation is applied while scoring.")
    parser.add_argument("--extra-train", default=None, choices=["asvspoof5"],
                        help="Score the run trained with this extra corpus (its own "
                             "checkpoint directory, plus-<name>/). Picks the checkpoint only.")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="An explicit checkpoint path, overriding --arch, "
                             "--frontend and --dp.")
    parser.add_argument("--dataset", default="asvspoof", choices=["asvspoof", "itw"],
                        help="'itw' scores In-the-Wild instead: 31,779 real-world "
                             "clips from 58 public figures. ASVspoof2019's attacks "
                             "are from 2019 and predate current voice cloning, so "
                             "the gap between the two is the generalisation result. "
                             "EER only — see in_the_wild.py for what it cannot "
                             "measure. --corpus and --partition are ignored.")
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

    # The directory layout is get_ckpt_paths' business, not ours — it is
    # imported rather than reimplemented so the two cannot drift apart.
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        frontend_sel = args.frontend or default_frontend_for(args.arch)
        _, _, ckpt_path = get_ckpt_paths(args.dp, frontend_sel, args.arch, args.rawboost,
                                         args.extra_train)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            f"Train one first:  python train_dp_avspoof.py --corpus {args.corpus} "
            f"--arch {args.arch} --no-dp")

    if DEVICE.type == "cuda":
        print(f"Device: cuda -> {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    else:
        print(f"Device: CPU (no CUDA visible), torch {torch.__version__}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Build the network this checkpoint came from, not whichever one is
    # currently the default. Checkpoints predating the architecture field are
    # the CNN, which is what DEFAULT_ARCH is.
    arch = ckpt.get("arch") or DEFAULT_ARCH
    model = build_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    # Score with the features this model was trained on, not a default. The
    # network accepts either channel count, so a mismatch is a wrong number
    # rather than a crash.
    frontend = ckpt.get("frontend") or DEFAULT_FRONTEND

    # The threshold the checkpoint carries was calibrated on dev. It is the one
    # the server actually applies, so it is the honest deployment operating
    # point — at serving time there are no eval labels to tune against.
    dev_threshold = ckpt.get("threshold")
    calibrated = dev_threshold is not None
    if not calibrated:
        dev_threshold = 0.5

    print(f"Checkpoint    {ckpt_path}")
    print(f"  epoch       {ckpt.get('epoch')}")
    print(f"  arch        {arch} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"  front-end   {frontend}")
    print(f"  regime      {'DP' if ckpt.get('dp') else 'non-private'}")
    # This epoch's own dev EER, not the run's running best. They differ the
    # moment an epoch is worse than its predecessor — LFCC epoch 5 scored 5.73%
    # against a running best of 5.34% — and conflating them silently flattens
    # the dev curve exactly where it turns, which is the part worth seeing.
    ckpt_metrics = ckpt.get("metrics") or {}
    dev_eer_epoch = ckpt_metrics.get("dev_eer")
    dev_eer_best = ckpt.get("best_eer")
    print(f"  dev EER     {fmt_pct(dev_eer_epoch)}  (this epoch)")
    if dev_eer_best is not None and dev_eer_epoch is not None and \
       abs(dev_eer_best - dev_eer_epoch) > 1e-9:
        print(f"              {fmt_pct(dev_eer_best)}  (best so far in the run)")
    print(f"  threshold   {dev_threshold:.4f} "
          f"({'calibrated on dev' if calibrated else 'DEFAULT 0.5, checkpoint carries none'})")

    if args.dataset == "itw":
        itw_root = get_itw_root()
        dataset = AVSpoofDataset(None, itw_root, build_transform(frontend),
                                 protocol=load_itw_protocol(itw_root), suffix="")
        paths, key = {}, None
        corpus_label, partition_label = "In-the-Wild", "all"
    else:
        paths = get_corpus_paths(args.corpus)
        key = args.partition.upper()
        if f"{key}_AUDIO_DIR" not in paths:
            raise FileNotFoundError(
                f"No {args.partition} partition under $ASVSPOOF_ROOT for {args.corpus}. "
                f"Expected ASVspoof2019_{args.corpus}_{args.partition}/flac and a matching protocol.")
        dataset = AVSpoofDataset(
            paths[f"{key}_PROTOCOL_FILE"], paths[f"{key}_AUDIO_DIR"], build_transform(frontend))
        corpus_label, partition_label = args.corpus, args.partition

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    counts = dataset.protocol["label"].map(LABEL_MAP).value_counts()
    print(f"\nDataset       {corpus_label} / {partition_label} ({len(dataset)} clips: "
          f"{int(counts.get(0, 0))} bonafide, {int(counts.get(1, 0))} spoof)")
    if args.dataset == "itw":
        print(f"Speakers      {dataset.protocol['speaker_id'].nunique()} public figures, "
              f"no attack taxonomy")
        print(f"NOTE          These are real-world deepfakes, not 2019 lab attacks.")
        print(f"              Expect a far worse number than LA eval — that is the finding.")
    else:
        print(f"Attacks       {', '.join(sorted(set(dataset.protocol['system_id'])))}")
    print(f"\nScoring with {args.num_workers} workers...")

    scores, labels = score_dataset(model, loader)
    system_ids = dataset.protocol["system_id"].to_numpy()

    # Two operating points, and the distance between them is the point.
    pooled_eer, eer_threshold = compute_eer_np(labels, scores)

    # min t-DCF is ASVspoof2019's PRIMARY metric; EER is the secondary one. It
    # is computed only when the organisers' ASV scores are present, because
    # they are what make the number comparable across systems — a t-DCF against
    # a different ASV is not the same quantity.
    min_tdcf, tdcf_detail = None, None
    asv_key = f"{key}_ASV_SCORES" if key else None
    if args.dataset == "itw":
        print("  (min t-DCF not defined here: it needs the organisers' ASV scores,")
        print("   which ship only with ASVspoof. A t-DCF against a different ASV is")
        print("   not the same quantity, so EER is the whole result.)")
    elif asv_key in paths:
        try:
            min_tdcf, tdcf_detail = compute_min_tdcf(
                scores[labels == 0], scores[labels == 1], paths[asv_key])
        except Exception as exc:                      # noqa: BLE001
            print(f"  (min t-DCF unavailable: {type(exc).__name__}: {exc})")
    else:
        print(f"  (min t-DCF skipped: no ASV score file for the {args.partition} partition)")
    cm_oracle, far_oracle, frr_oracle = rates_at(labels, scores, eer_threshold)
    cm_dev, far_dev, frr_dev = rates_at(labels, scores, dev_threshold)
    acc_dev = (cm_dev["tn"] + cm_dev["tp"]) / max(1, len(labels))

    print(f"\n=== {corpus_label} {partition_label} ===")
    if min_tdcf is not None:
        print(f"min t-DCF                {min_tdcf:6.4f}  <- ASVspoof2019 PRIMARY metric")
        print(f"                                 (1.0 = the 'accept everything' floor;")
        print(f"                                  AASIST reports 0.0275 on this partition)")
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

    # In-the-Wild carries no attack taxonomy, so every spoof row is "-" and the
    # per-attack table would just restate the pooled EER under a heading that
    # implies a breakdown exists. Suppress it rather than print a fake one.
    by_attack = per_attack_eer(labels, scores, system_ids)
    if set(by_attack) == {"-"}:
        by_attack = {}
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
        "dataset": args.dataset,
        "corpus": corpus_label,
        "partition": partition_label,
        # Kept so a per-speaker analysis of In-the-Wild needs no re-run: it is
        # the only grouping this dataset has, standing in for the attack IDs.
        "speakers": (dataset.protocol["speaker_id"].tolist()
                     if args.dataset == "itw" else None),
        "regime": "dp" if ckpt.get("dp") else "non-private",
        "arch": arch,
        "frontend": frontend,
        "epoch": ckpt.get("epoch"),
        "n_clips": int(len(labels)),
        "class_names": CLASS_NAMES,
        "dev": {"eer": dev_eer_epoch, "best_eer_in_run": dev_eer_best,
                "threshold": dev_threshold, "threshold_calibrated": calibrated},
        "pooled": {
            "min_tdcf": min_tdcf, "tdcf_detail": tdcf_detail,
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
    tag = "itw" if args.dataset == "itw" else args.partition
    out = args.out or ckpt_path.parent / f"eval_{tag}_{strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
