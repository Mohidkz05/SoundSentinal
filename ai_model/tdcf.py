# tdcf.py
#
# The minimum tandem detection cost function, t-DCF.
#
# WHY THIS EXISTS: min t-DCF is the *primary* metric of the ASVspoof2019
# challenge and EER is explicitly the secondary one. Every paper in the
# comparison table in APPROACH.md quotes both — AASIST reports 0.83% EER and
# 0.0275 min t-DCF — so reporting EER alone leaves our rows only half
# comparable with the rows we are measuring ourselves against.
#
# WHAT IT MEASURES that EER does not: a countermeasure does not run alone, it
# sits in front of a speaker verification system. t-DCF scores the pair. A
# countermeasure that rejects spoofs an ASV would have rejected anyway has
# bought nothing, and one that passes spoofs an ASV finds convincing has done
# real damage. EER treats every error as equally costly and ignores the ASV
# entirely; t-DCF weights errors by what they cost downstream.
#
# PROVENANCE: the formulas are the ASVspoof2019 organisers' reference
# implementation, as distributed with the evaluation package and carried in
# clovaai/aasist (MIT, Copyright (c) 2021-present NAVER Corp.). Reimplemented
# here rather than vendored so it runs on current numpy — the reference uses
# `np.float`, removed in numpy 1.24 — but the arithmetic is theirs and the
# numbers are meant to match. See:
#   https://github.com/clovaai/aasist/blob/main/evaluation.py
#   https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf

import numpy as np

# The ASVspoof2019 cost model, fixed by the evaluation plan. These are not
# tuning parameters — changing them makes the result incomparable to every
# published number.
COST_MODEL = {
    "Pspoof": 0.05,                 # prior probability of a spoofing attack
    "Ptar": (1 - 0.05) * 0.99,      # prior of a target trial   = 0.9405
    "Pnon": (1 - 0.05) * 0.01,      # prior of a nontarget trial = 0.0095
    "Cmiss_asv": 1,                 # cost: ASV rejects a target
    "Cfa_asv": 10,                  # cost: ASV accepts a nontarget
    "Cmiss_cm": 1,                  # cost: CM rejects bonafide
    "Cfa_cm": 10,                   # cost: CM accepts a spoof
}


def compute_det_curve(target_scores, nontarget_scores):
    """False-reject and false-accept rates across every threshold.

    "target" is the class scored high. For a countermeasure that is *bonafide*,
    which is the opposite orientation to the rest of this repo — our model
    emits P(spoof), where high means spoof. compute_metrics() below does the
    flip, in one place, so nothing else has to think about it.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))

    # mergesort for a stable order, so ties resolve identically to the
    # reference implementation and the numbers reproduce exactly.
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """EER and the threshold it occurs at, from the DET curve."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    i = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[i], far[i]))), float(thresholds[i])


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """The three ASV rates t-DCF needs, at the ASV's own operating point."""
    pfa_asv = float(np.sum(non_asv >= asv_threshold) / non_asv.size)
    pmiss_asv = float(np.sum(tar_asv < asv_threshold) / tar_asv.size)
    pmiss_spoof_asv = (None if spoof_asv.size == 0
                       else float(np.sum(spoof_asv < asv_threshold) / spoof_asv.size))
    return pfa_asv, pmiss_asv, pmiss_spoof_asv


def load_asv_scores(path):
    """Read the organisers' ASV score file: `<source> <key> <score>` per line.

    The keys are target / nontarget / spoof. These scores are supplied with the
    corpus and are not ours to regenerate — the whole point of t-DCF is that
    every system is scored against the *same* ASV.
    """
    data = np.genfromtxt(str(path), dtype=str)
    keys, scores = data[:, 1], data[:, 2].astype(float)
    return (scores[keys == "target"],
            scores[keys == "nontarget"],
            scores[keys == "spoof"])


def compute_min_tdcf(p_spoof_bonafide, p_spoof_attack, asv_scores_path,
                     cost_model=COST_MODEL):
    """Minimum normalised t-DCF for a countermeasure.

    Takes this repo's native orientation — P(spoof) for bonafide clips and for
    spoofed clips — and flips it internally to the bonafide-high convention the
    reference implementation uses.

    Returns (min_tdcf, detail). A normalised value of 1.0 is the floor of
    usefulness: it is what you would score by accepting everything, so anything
    at or above 1.0 means the countermeasure is not contributing.
    """
    tar_asv, non_asv, spoof_asv = load_asv_scores(asv_scores_path)
    _, asv_threshold = compute_eer(tar_asv, non_asv)
    pfa_asv, pmiss_asv, pmiss_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)

    # Flip to bonafide-high. Only the ordering matters to a threshold sweep,
    # so any decreasing transform gives the same curve; 1 - p keeps the value
    # readable as "probability this is real".
    bonafide_cm = 1.0 - np.asarray(p_spoof_bonafide, dtype=float)
    spoof_cm = 1.0 - np.asarray(p_spoof_attack, dtype=float)

    pmiss_cm, pfa_cm, _ = compute_det_curve(bonafide_cm, spoof_cm)

    c1 = (cost_model["Ptar"] * (cost_model["Cmiss_cm"]
                                - cost_model["Cmiss_asv"] * pmiss_asv)
          - cost_model["Pnon"] * cost_model["Cfa_asv"] * pfa_asv)
    c2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - pmiss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError(
            f"Negative t-DCF weights (C1={c1}, C2={c2}). The ASV error rates are "
            f"wrong — check that the ASV score file matches the partition being scored.")

    tdcf = c1 * pmiss_cm + c2 * pfa_cm
    tdcf_norm = tdcf / min(c1, c2)
    i = int(np.argmin(tdcf_norm))
    return float(tdcf_norm[i]), {
        "asv_threshold": asv_threshold,
        "Pfa_asv": pfa_asv, "Pmiss_asv": pmiss_asv,
        "Pmiss_spoof_asv": pmiss_spoof_asv,
        "C1": float(c1), "C2": float(c2),
    }
