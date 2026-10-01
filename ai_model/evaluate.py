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
from torch.utils.data import DataLoader

from model import (ARCHITECTURES, CLASS_NAMES, DEFAULT_ARCH, DEFAULT_FRONTEND,
                   FRONTENDS, LABEL_MAP, build_model, build_transform,
                   default_frontend_for)
from in_the_wild import get_itw_root, load_protocol as load_itw_protocol
from tdcf import compute_min_tdcf
from train_dp_avspoof import (
    AVSpoofDataset,
    BONAFIDE_MIXES,
    EXTRA_TRAIN_MIXES,
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


def prob_to_log_odds(p):
    """P(spoof) threshold -> the log-odds scale score_dataset() returns."""
    with np.errstate(divide="ignore"):
        return float(np.log(p) - np.log1p(-p))


def log_odds_to_prob(s):
    return float(1.0 / (1.0 + np.exp(-s)))


@torch.no_grad()
def score_dataset(model, loader, with_cleanliness=False):
    """Run the model over a loader, returning log-odds of spoof and the labels.

    Log-odds, not P(spoof). The two rank clips identically — P(spoof) is the
    sigmoid of this — until softmax saturates: in float32 every clip past a
    log-odds of ~17 is exactly 1.0. SSL-AASIST is confident enough that 15,307
    In-the-Wild clips tied at 1.0, and an EER computed over a tie that size is
    an arbitrary point inside it (it reported 37.82% at "threshold 1.0000").
    The logit difference never saturates, so every EER and min t-DCF here is
    computed on it and thresholds are converted, not the other way round.

    With `with_cleanliness` the loader's dataset must be built with
    with_cleanliness=True, and a third array, model.cleanliness_db per clip, is
    returned too.
    """
    model.eval()
    all_scores, all_labels, all_clean = [], [], []
    total = len(loader.dataset)
    seen = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        if with_cleanliness:
            all_clean.append(batch[2])
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x).float()
        all_scores.append((logits[:, 1] - logits[:, 0]).cpu())
        all_labels.append(y)
        seen += y.size(0)
        # A plain periodic line rather than a progress bar: this runs in a Slurm
        # job where stdout is a file, and a bar would write one line per update.
        if seen % (BATCH_SIZE * 50) == 0 or seen == total:
            print(f"  scored {seen}/{total}", flush=True)
    if with_cleanliness:
        return (torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy(),
                torch.cat(all_clean).numpy())
    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def per_attack_eer(labels, scores, system_ids, served_threshold=None):
    """EER for each attack on its own, each against the full bonafide set.

    This is how ASVspoof papers break results down, and it is the part worth
    reading: a single pooled EER hides that a detector can be near-perfect on
    most attacks and blind to one. A03 at 40% and everything else at 2% is a
    completely different finding from a uniform 5%, and only this table
    distinguishes them.

    With `served_threshold` (log-odds, a scalar or one value per clip when a
    clean-audio route applies), each attack also gets the share of its fakes
    that score below it, i.e. pass at the threshold the server applies.
    EER says whether the model can separate an attack; this says whether the
    line it is actually served at does (RESULTS.md Finding 12).
    """
    bonafide = labels == 0
    out = OrderedDict()
    for attack in sorted(set(system_ids[labels == 1])):
        is_attack = (labels == 1) & (system_ids == attack)
        mask = bonafide | is_attack
        eer, thr = compute_eer_np(labels[mask], scores[mask])
        out[str(attack)] = {
            "eer": eer,
            "threshold": thr,
            "n_spoof": int(is_attack.sum()),
        }
        if served_threshold is not None:
            thr = (served_threshold[is_attack] if np.ndim(served_threshold)
                   else served_threshold)
            out[str(attack)]["passed_at_served"] = float((scores[is_attack] < thr).mean())
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
    parser.add_argument("--extra-train", default=None, choices=list(EXTRA_TRAIN_MIXES),
                        help="Score the run trained with this extra corpus (its own "
                             "checkpoint directory, plus-<name>/). Picks the checkpoint only.")
    parser.add_argument("--extra-bonafide", default=None, choices=list(BONAFIDE_MIXES),
                        help="Score the run trained with this extra bona fide speech "
                             "(plus-<name>-bonafide/). Picks the checkpoint only.")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="An explicit checkpoint path, overriding --arch, "
                             "--frontend and --dp.")
    parser.add_argument("--dataset", default="asvspoof",
                        choices=["asvspoof", "itw", "speechfake", "synth"],
                        help="'itw' scores In-the-Wild instead: 31,779 real-world "
                             "clips from 58 public figures. ASVspoof2019's attacks "
                             "are from 2019 and predate current voice cloning, so "
                             "the gap between the two is the generalisation result. "
                             "EER only — see in_the_wild.py for what it cannot "
                             "measure. --corpus and --partition are ignored. "
                             "'speechfake' scores SpeechFake-BD's baseline test "
                             "split: clean, modern TTS/VC/vocoder output, from the "
                             "SAME 30 generators its train split holds, so for a "
                             "model trained with --extra-train speechfake it is a "
                             "seen-generator test and an optimistic one.")
    parser.add_argument("--language", default="en", choices=["en", "zh", "all"],
                        help="--dataset speechfake only: which language's rows to "
                             "score. English matches In-the-Wild and the app.")
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
                                         args.extra_train, args.extra_bonafide)
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
    # calibrate.py records the input degradation its threshold was fitted
    # under; scoring without it would pair that threshold with other inputs.
    degradation = ckpt.get("input_degradation") or "none"
    # calibrate_clean.py's clean-audio route (Finding 15): clips at or above
    # the cleanliness cutoff are held to a second threshold.
    routing = ckpt.get("routing")

    # The threshold the checkpoint carries was calibrated on dev. It is the one
    # the server actually applies, so it is the honest deployment operating
    # point — at serving time there are no eval labels to tune against.
    dev_threshold = ckpt.get("threshold")
    calibrated = dev_threshold is not None
    # calibrate.py rewrites the threshold and records where it came from.
    threshold_source = ckpt.get("threshold_source") or (
        "calibrated on dev" if calibrated else "DEFAULT 0.5, checkpoint carries none")
    if not calibrated:
        dev_threshold = 0.5

    print(f"Checkpoint    {ckpt_path}")
    print(f"  epoch       {ckpt.get('epoch')}")
    print(f"  arch        {arch} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"  front-end   {frontend}")
    print(f"  degradation {degradation}")
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
          f"({threshold_source})")
    if routing:
        print(f"  clean route log-odds {routing['threshold_clean_log_odds']:+.2f} for clips "
              f">= {routing['cutoff_db']:.1f} dB dynamic range ({routing['source']})")

    if args.dataset == "itw":
        itw_root = get_itw_root()
        dataset = AVSpoofDataset(None, itw_root, build_transform(frontend),
                                 protocol=load_itw_protocol(itw_root), suffix="",
                                 degradation=degradation)
        paths, key = {}, None
        corpus_label, partition_label = "In-the-Wild", "all"
    elif args.dataset == "speechfake":
        import speechfake
        # --partition eval means SpeechFake's test split; dev is its dev split,
        # which Finding 14 compares candidates on so test is read once.
        sf_part = "test" if args.partition == "eval" else "dev"
        frame, sf_root = speechfake.load_protocol(sf_part)
        if args.language != "all":
            # load_protocol drops the language column; read it back from the
            # same CSV, row for row.
            meta = speechfake.read_metadata(sf_part, sf_root)
            frame = frame[(meta["language"] == args.language).to_numpy()].reset_index(drop=True)
        dataset = AVSpoofDataset(None, sf_root, build_transform(frontend),
                                 protocol=frame, suffix="", degradation=degradation)
        paths, key = {}, None
        corpus_label, partition_label = "SpeechFake-BD", f"{sf_part} ({args.language})"
    elif args.dataset == "synth":
        import synth
        # Our own fakes from held-out generator families, never trained on,
        # with the same LibriSpeech test-clean speakers' genuine clips as the
        # real side (Finding 18). --partition dev is heldout-a (Stage A), eval
        # is heldout-b (Stage B, read once).
        split = "heldout-b" if args.partition == "eval" else "heldout-a"
        frame, root = synth.load_heldout(split)
        dataset = AVSpoofDataset(None, root, build_transform(frontend),
                                 protocol=frame, suffix="", degradation=degradation)
        paths, key = {}, None
        corpus_label, partition_label = "Own fakes, held-out families", split
    else:
        paths = get_corpus_paths(args.corpus)
        key = args.partition.upper()
        if f"{key}_AUDIO_DIR" not in paths:
            raise FileNotFoundError(
                f"No {args.partition} partition under $ASVSPOOF_ROOT for {args.corpus}. "
                f"Expected ASVspoof2019_{args.corpus}_{args.partition}/flac and a matching protocol.")
        dataset = AVSpoofDataset(
            paths[f"{key}_PROTOCOL_FILE"], paths[f"{key}_AUDIO_DIR"], build_transform(frontend),
            degradation=degradation)
        corpus_label, partition_label = args.corpus, args.partition

    dataset.with_cleanliness = routing is not None
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

    noisy_threshold = prob_to_log_odds(dev_threshold)
    if routing:
        scores, labels, cleanliness = score_dataset(model, loader, with_cleanliness=True)
        routed_clean = cleanliness >= routing["cutoff_db"]
        served_threshold = np.where(routed_clean, routing["threshold_clean_log_odds"],
                                    noisy_threshold)
    else:
        scores, labels = score_dataset(model, loader)
        cleanliness, routed_clean, served_threshold = None, None, noisy_threshold
    system_ids = dataset.protocol["system_id"].to_numpy()

    # Two operating points, and the distance between them is the point.
    pooled_eer, eer_threshold = compute_eer_np(labels, scores)

    # min t-DCF is ASVspoof2019's PRIMARY metric; EER is the secondary one. It
    # is computed only when the organisers' ASV scores are present, because
    # they are what make the number comparable across systems — a t-DCF against
    # a different ASV is not the same quantity.
    min_tdcf, tdcf_detail = None, None
    asv_key = f"{key}_ASV_SCORES" if key else None
    if args.dataset in ("itw", "speechfake", "synth"):
        print("  (min t-DCF not defined here: it needs the organisers' ASV scores,")
        print("   which ship only with ASVspoof. A t-DCF against a different ASV is")
        print("   not the same quantity, so EER is the whole result.)")
    elif asv_key in paths:
        try:
            # Log-odds, though tdcf.py documents P(spoof): it only flips and
            # sweeps the scores, so any order-preserving scale gives the same
            # minimum.
            min_tdcf, tdcf_detail = compute_min_tdcf(
                scores[labels == 0], scores[labels == 1], paths[asv_key])
        except Exception as exc:                      # noqa: BLE001
            print(f"  (min t-DCF unavailable: {type(exc).__name__}: {exc})")
    else:
        print(f"  (min t-DCF skipped: no ASV score file for the {args.partition} partition)")
    cm_oracle, far_oracle, frr_oracle = rates_at(labels, scores, eer_threshold)
    cm_dev, far_dev, frr_dev = rates_at(labels, scores, served_threshold)
    eer_threshold_prob = log_odds_to_prob(eer_threshold)
    acc_dev = (cm_dev["tn"] + cm_dev["tp"]) / max(1, len(labels))

    print(f"\n=== {corpus_label} {partition_label} ===")
    if min_tdcf is not None:
        print(f"min t-DCF                {min_tdcf:6.4f}  <- ASVspoof2019 PRIMARY metric")
        print(f"                                 (1.0 = the 'accept everything' floor;")
        print(f"                                  AASIST reports 0.0275 on this partition)")
    print(f"EER                      {pooled_eer*100:6.2f}%  (at its own threshold P={eer_threshold_prob:.4g}, "
          f"log-odds {eer_threshold:+.2f})")
    print(f"  confusion @EER         bonafide {cm_oracle['tn']} ok / {cm_oracle['fp']} flagged | "
          f"spoof {cm_oracle['tp']} caught / {cm_oracle['fn']} missed")
    print(f"\nAt the served threshold {dev_threshold:.4f} ({threshold_source}):")
    print(f"  accuracy               {acc_dev*100:6.2f}%")
    print(f"  false accept (spoof passed)   {frr_dev*100:6.2f}%")
    print(f"  false reject (real flagged)   {far_dev*100:6.2f}%")
    print(f"  confusion              bonafide {cm_dev['tn']} ok / {cm_dev['fp']} flagged | "
          f"spoof {cm_dev['tp']} caught / {cm_dev['fn']} missed")
    routing_result = None
    if routing:
        cm_one, far_one, frr_one = rates_at(labels, scores, noisy_threshold)
        real, fake = labels == 0, labels == 1
        routing_result = {
            "cutoff_db": routing["cutoff_db"],
            "threshold_clean_log_odds": routing["threshold_clean_log_odds"],
            "routed_clean_real": float(routed_clean[real].mean()) if real.any() else None,
            "routed_clean_fake": float(routed_clean[fake].mean()) if fake.any() else None,
            "at_single_threshold": {"false_accept_rate": frr_one, "false_reject_rate": far_one,
                                    "confusion": cm_one},
            "cleanliness_db_percentiles": {
                str(q): float(np.percentile(cleanliness, q)) for q in (5, 25, 50, 75, 95)},
        }
        print(f"  (routed: {routing_result['routed_clean_real'] or 0:.1%} of real and "
              f"{routing_result['routed_clean_fake'] or 0:.1%} of fake clips took the clean route)")
        print(f"  single threshold, for comparison: spoof passed {frr_one*100:6.2f}%, "
              f"real flagged {far_one*100:6.2f}%")
    print("\n  The distance between these two blocks is the cost of calibrating on dev")
    print("  and deploying against unseen attacks. The EER line is what compares to")
    print("  published numbers; the threshold block is what a user would experience.")

    # In-the-Wild carries no attack taxonomy, so every spoof row is "-" and the
    # per-attack table would just restate the pooled EER under a heading that
    # implies a breakdown exists. Suppress it rather than print a fake one.
    by_attack = per_attack_eer(labels, scores, system_ids, served_threshold)
    if set(by_attack) == {"-"}:
        by_attack = {}
    if by_attack:
        print(f"\n=== By attack (EER against all bonafide; share passed at the served threshold) ===")
        worst = max(by_attack.items(), key=lambda kv: kv[1]["eer"])
        best = min(by_attack.items(), key=lambda kv: kv[1]["eer"])
        width = max(6, *(len(a) for a in by_attack))
        for attack, r in by_attack.items():
            mark = "  <- worst" if attack == worst[0] else ("  <- best" if attack == best[0] else "")
            print(f"  {attack:<{width}} EER {r['eer']*100:6.2f}%   passed {r['passed_at_served']*100:6.2f}%"
                  f"   n={r['n_spoof']:<6}{mark}")
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
        "input_degradation": degradation,
        "routing": routing_result,
        "epoch": ckpt.get("epoch"),
        "n_clips": int(len(labels)),
        "class_names": CLASS_NAMES,
        "dev": {"eer": dev_eer_epoch, "best_eer_in_run": dev_eer_best,
                "threshold": dev_threshold, "threshold_calibrated": calibrated},
        "pooled": {
            "min_tdcf": min_tdcf, "tdcf_detail": tdcf_detail,
            "eer": pooled_eer, "eer_threshold": eer_threshold_prob,
            "eer_threshold_log_odds": eer_threshold,
            "score_scale": "log-odds",
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
    tag = {"itw": "itw",
           "speechfake": f"speechfake-{args.partition}-{args.language}",
           "synth": f"synth-{args.partition}"}.get(args.dataset,
                                                                            args.partition)
    out = args.out or ckpt_path.parent / f"eval_{tag}_{strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
