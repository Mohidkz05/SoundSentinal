# SoundSentinal — experimental results

Every number produced against the current architecture, with the run that
produced it. `APPROACH.md` records *what we chose and why*; this file records
*what happened*. When the two disagree about a number, this file is newer.

**All runs: 20–21 September 2026, Monash M3 (project `df37`), trained on
ASVspoof2019 LA.** CNN runs are 5 epochs, batch 64, Adam at 1e-3,
inverse-frequency class weights `[4.919, 0.557]`, seed 42. The AASIST run uses
its published recipe instead — 100 epochs, batch 24, Adam at 1e-4 with weight
decay and cosine annealing — so it differs from the CNN rows by schedule as
well as architecture (Finding 5).

## How to read these numbers

Three rules, each learned the hard way and each easy to undo by accident.

1. **Eval, never dev.** The dev partition reuses attacks A01–A06, the six seen
   in training. Eval holds A07–A19, unseen. Our log-Mel baseline scored 0.24% on
   dev and 10.15% on eval — a factor of 42. Any dev figure quoted beside a
   published one is comparing different quantities.
2. **min t-DCF is the primary metric**, EER the secondary one, per the
   ASVspoof2019 evaluation plan. Normalised, 1.0 is the "accept everything"
   floor.
3. **Pooled figures hide the finding.** Every system here has an EER spread of
   two orders of magnitude across attacks. The per-attack tables are the real
   result; the pooled number is an average over behaviours that have nothing to
   do with each other.

## Summary

Reference rows are from Table 8 (evaluation set) of the ASVspoof2019 database
paper. Ours are measured here; raw JSON lives beside each checkpoint in
`$CKPT_ROOT`, written by `evaluate.py`.

| System | Front-end | Regime | eval EER | min t-DCF |
| --- | --- | --- | --- | --- |
| AASIST (published, **not ours**) | raw waveform | non-private | 0.83% | 0.0275 |
| **Our AASIST, `best.pth` (epoch 42)** | raw waveform | non-private | **3.17%** | **0.0909** |
| LFCC-GMM (official B2) | LFCC | non-private | 8.09% | 0.2116 |
| **Our CNN, epoch 2** | log-Mel | non-private | **9.60%** | **0.2124** |
| CQCC-GMM (official B1) | CQCC | non-private | 9.57% | 0.2366 |
| Our CNN, `best.pth` | log-Mel | non-private | 10.15% | 0.2350 |
| Our CNN, epoch 5 | LFCC | non-private | 13.10% | 0.2503 |
| Our CNN, `best.pth` | LFCC | non-private | 13.72% | 0.2463 |
| **Our CNN, epoch 3** | log-Mel | **DP, ε=0.48** | **17.57%** | **0.2609** |
| Our CNN, `best.pth` | log-Mel | DP, ε=0.48 | 17.80% | 0.2696 |

Our AASIST's best single epoch on eval reaches 2.98% EER (epochs 59, 64) and
0.0807 min t-DCF (epochs 85, 95), but those are picked by looking at eval, so
the `best.pth` row is the one to quote. See Finding 5.

## Finding 1 — dev EER does not select the best model

Job 60256177 trained; job 60261175 scored every epoch it saved.

| Epoch | dev EER | eval EER | min t-DCF | |
| --- | --- | --- | --- | --- |
| 1 | 6.75% | 15.45% | 0.2758 | |
| 2 | 1.49% | **9.60%** | **0.2124** | best on eval |
| 3 | 0.74% | 10.94% | 0.2536 | |
| 4 | 0.48% | 11.48% | 0.2909 | |
| 5 | 0.24% | 10.15% | 0.2350 | what `best.pth` selected |

**Dev EER improves monotonically and eval EER does not.** After epoch 2, dev
improves six-fold while eval gets *worse*. `save_ckpt` selects `best.pth` by dev
EER, so it shipped epoch 5 — and epoch 2 is better on both eval metrics.

The cost is not academic. Epoch 2 at 9.60% / 0.2124 beats CQCC-GMM on min t-DCF
and effectively ties LFCC-GMM on it (0.2116). Selecting by dev turned a result
competitive with both official baselines into one that loses to both.

Epochs 3–5 are the model learning the six training attacks better and
generalising worse — textbook overfitting, invisible to the criterion being
used to stop.

**A note on the correlation statistic**, because `summarise_results.py` prints
one and it is the wrong summary here. Dev and eval correlate at +0.917 across
these epochs, which sounds reassuring and is not: the coefficient is dominated
by epoch 1, where both are simply bad. What matters is the *rank*, and the rank
is wrong — the best dev epoch is the fourth-best eval epoch. A high correlation
and a broken selection criterion coexist comfortably.

**Consequence:** model selection needs a held-out set containing unseen attack
types. Until it has one, every comparison in this file is between arbitrary
epochs rather than between best-available models, and that caveat belongs in
any writeup quoting them.

## Finding 2 — the two front-ends are complementary, not ranked

Jobs 60261176 (train) and 60261216 (eval). Same network, same schedule, same
data; only `build_transform()` differs. Pooled, LFCC looks worse: 13.72%
against 10.15%. Per attack it is a different story.

| Attack | log-Mel | LFCC | |
| --- | --- | --- | --- |
| A07 | 0.11% | 0.53% | |
| A08 | 0.83% | 6.47% | log-Mel better |
| A09 | 0.63% | 0.22% | |
| A10 | 3.70% | **0.61%** | LFCC 6× better |
| A11 | 2.97% | **0.43%** | LFCC 7× better |
| A12 | 8.10% | **0.61%** | LFCC 13× better |
| A13 | 15.85% | **0.87%** | LFCC 18× better |
| A14 | 2.12% | **0.53%** | LFCC 4× better |
| A15 | 5.72% | **0.34%** | LFCC 17× better |
| A16 | 0.22% | 1.10% | |
| A17 | 41.19% | 33.76% | both poor |
| A18 | 12.19% | **41.29%** | log-Mel 3.4× better |
| A19 | 2.61% | **25.03%** | log-Mel 10× better |
| **Pooled** | **10.15%** | **13.72%** | |

**LFCC wins nine of thirteen attacks, several by more than an order of
magnitude, and loses the pooled number on A18 and A19 alone.** Neither front-end
dominates. The hypothesis in `APPROACH.md` — that the Mel scale compresses the
high frequencies where synthesis artefacts live — is confirmed for A10–A15 and
falsified for A18–A19, where log-Mel sees something LFCC does not.

**A corroboration worth noting:** the official LFCC-GMM baseline is also much
worse on A19 than the CQCC one (13.94% against 0.04%). Our LFCC front-end
reproduces a documented weakness of LFCC features, which is evidence the
implementation is behaving like a real LFCC system rather than a broken one.

**Consequence, and it is the strongest argument in this file for AASIST:** the
answer is not to pick the better handcrafted front-end, because there isn't one.
It is either to fuse them — standard practice in the ASVspoof literature, and
the per-attack minimum of these two columns would be a strong system — or to
stop handcrafting and let the model learn its filterbank, which is precisely
what AASIST's SincNet front-end does.

**LFCC was also still improving at epoch 5** (dev 5.34% → 5.73%, eval 13.72% →
13.10%), so 5 epochs under-trains it, while log-Mel had begun overfitting by
epoch 2. The two front-ends do not want the same schedule, and a fair comparison
has to account for that.

**LFCC generalises far more honestly.** Its dev→eval gap is 5.34% → 13.10%, a
factor of 2.5, against log-Mel's factor of 42. It is not memorising the training
attacks to anything like the same degree.

## Finding 3 — the first cost-of-privacy measurement

Jobs 60261177 (train) and 60261217 (eval). Identical to the log-Mel baseline
except Opacus is engaged: `noise_multiplier=1.1`, `max_grad_norm=1.0`,
δ=1e-5, reaching **ε=0.48** after 5 epochs.

Comparing each regime's `best.pth`:

| | non-private | DP (ε=0.48) | cost |
| --- | --- | --- | --- |
| eval EER | 10.15% | 17.80% | **+7.65 points** |
| min t-DCF | 0.2350 | 0.2696 | +0.0346 |
| dev EER | 0.24% | 24.41% | +24.17 points |

Comparing each regime's best epoch on eval, which is the fairer reading:

| | non-private (ep 2) | DP (ep 3) | cost |
| --- | --- | --- | --- |
| eval EER | 9.60% | 17.57% | **+7.97 points** |
| min t-DCF | 0.2124 | 0.2609 | +0.0485 |

Per epoch, job 60261275:

| Epoch | dev EER | eval EER | min t-DCF |
| --- | --- | --- | --- |
| 1 | 35.48% | 22.19% | 0.4677 |
| 2 | 31.68% | 17.70% | 0.2714 |
| 3 | 29.99% | **17.57%** | **0.2609** |
| 4 | 28.15% | 17.87% | 0.2651 |
| 5 | 24.41% | 17.80% | 0.2696 |

**The DP model scores better on eval than on dev** — 17.80% against 24.41% —
which is the reverse of every non-private run here and worth explaining rather
than glossing. Under DP noise the model never learned the six training attacks
sharply enough to be flattered by dev, so dev stopped being an optimistic
estimate. Its eval EER also flattens after epoch 2 while dev keeps falling: the
same divergence as Finding 1, at a different scale.

This is the project's central question getting its first number. Read it with
three caveats, all of which understate or distort the cost:

- **Only five epochs, and eval had plateaued by epoch 2** while ε kept being
  spent. The privacy budget bought nothing after that, which is an argument
  about schedule rather than about DP.
- **The DP run's hyperparameters are untuned.** DP-SGD generally wants a larger
  batch and a different learning rate; running it with the non-private settings
  minus a flag measures *DP with the wrong hyperparameters*, which overstates
  the cost of privacy as such.
- **ε=0.48 is a very strong guarantee**, far stricter than the ε≈8 commonly
  reported in DP deep learning. Part of this gap is the strictness of the
  privacy level, not privacy in principle. A sweep over noise multipliers would
  turn one point into the curve the research question actually asks for.
- **Neither arm is the best-available model**, per Finding 1.

**A failure mode to watch:** the DP model's dev accuracy sat at exactly 89.74%
for all five epochs — the proportion of spoof in the dev partition. Under DP
noise it drifted toward predicting the majority class, which is the exact
pathology the class weighting was added to prevent. EER at 24.41% shows it is
not *purely* a majority-class predictor, but accuracy is doing no work here.

Per attack, DP is excellent on A07–A16 (0.39%–1.08%, several *better* than the
non-private log-Mel run) and effectively blind on A17 (43.22%), A18 (44.53%) and
A19 (44.39%) — all close to chance. DP noise did not degrade the model evenly;
it removed three attack families.

## Finding 4 — the calibrated threshold does not survive the partition change

The operating point stored in `best.pth` is calibrated on dev. Applied to eval:

| | log-Mel non-private | DP |
| --- | --- | --- |
| threshold | 0.5698 | 0.9927 |
| eval EER threshold | 0.0070 | 0.9932 |
| accuracy at served threshold | 70.78% | 82.64% |
| **spoofs passed** | **32.51%** | 16.99% |
| real clips flagged | 0.61% | 20.54% |

For the non-private model the served threshold is nearly two orders of magnitude
from where eval's EER point sits, and a third of deepfakes get through. This has
product consequences beyond the research: `/result` draws this threshold on
screen and describes each reading relative to it.

The threshold also moved erratically during training — 0.5030, 0.6735, 0.3342,
0.8935, 0.5698 — so it is unstable across epochs as well as across partitions.

**Consequence:** calibration needs a held-out set whose attacks are unseen, for
the same reason model selection does. Calibrating on eval would be test-set
leakage and is not an option.

## Finding 5 — AASIST, measured: three times better than the CNN, four times worse than the paper

Jobs 60271182 (train, 100 epochs, about 11h45m on one L40S), 60287185 (eval of
`best.pth`) and 60287251 (every epoch scored on eval). Non-private, AASIST's
published recipe: batch 24, Adam at 1e-4, weight decay, cosine annealing.

| | eval EER | min t-DCF |
| --- | --- | --- |
| Our best CNN (log-Mel, epoch 2) | 9.60% | 0.2124 |
| **Our AASIST, `best.pth` (epoch 42)** | **3.17%** | **0.0909** |
| AASIST as published | 0.83% | 0.0275 |

**It beats both official GMM baselines and every CNN run by a wide margin**:
min t-DCF 0.0909 against 0.2116 for LFCC-GMM, less than half. It is still about
3.3× off the paper on both metrics. Likely reasons, none tested yet:

- **GroupNorm instead of BatchNorm** — the change that keeps the model
  DP-compatible (see "The port" in `APPROACH.md`). The paper's number comes
  from BatchNorm.
- **One seed.** Adjacent epochs here already swing by nearly 2 points of eval
  EER, so a single run cannot say how much of the gap is luck. Before quoting
  the paper's 0.83% as a target, check whether it is a best-of-several-runs
  figure.
- **The epoch was chosen by dev EER**, which Finding 1 showed to be unreliable.

**Model selection hurts less here than on the CNN, but it still picks
imperfectly.** Dev EER bottomed out at 0.98% on epoch 42, so that became
`best.pth`. On eval, epoch 42 ranks 23rd of 100 by EER and 29th by min t-DCF.
The best epoch is only 0.19 points better on EER (2.98%) and 0.010 better on
t-DCF (0.0807). After about epoch 40 eval EER stays within a narrow band
(2.98%–5.00%, mean 3.30%), so here the choice of epoch matters much less than
it did for the CNN.

The summariser prints a dev/eval correlation of +0.985 for this sweep. As in
Finding 1, that number flatters the selection rule, because epochs 1–9 are
bad on both partitions. Measured from epoch 10 onward, the rank correlation
drops to +0.62: dev tells you roughly where you are, not which epoch is best.

**Per attack, AASIST closes the gaps the two handcrafted front-ends left** (see
Finding 2):

| Attack | log-Mel CNN | LFCC CNN | AASIST |
| --- | --- | --- | --- |
| A07 | 0.11% | 0.53% | 0.61% |
| A08 | 0.83% | 6.47% | 1.69% |
| A09 | 0.63% | 0.22% | 0.39% |
| A10 | 3.70% | 0.61% | 0.91% |
| A11 | 2.97% | 0.43% | 0.55% |
| A12 | 8.10% | 0.61% | 0.55% |
| A13 | 15.85% | 0.87% | 0.55% |
| A14 | 2.12% | 0.53% | 0.55% |
| A15 | 5.72% | 0.34% | 2.23% |
| A16 | 0.22% | 1.10% | 1.71% |
| A17 | 41.19% | 33.76% | **3.90%** |
| A18 | 12.19% | 41.29% | **10.48%** |
| A19 | 2.61% | 25.03% | **2.09%** |
| **Pooled** | 10.15% | 13.72% | **3.17%** |

A17, which defeated both CNNs, falls from over 33% to 3.90%. A18 is still the
hardest attack by far, at 10.48%, and alone accounts for most of the pooled
figure. AASIST does not win every row, so it has not learned a strictly better
version of either front-end, but it has no near-chance attack left, which
neither CNN managed. That is the argument Finding 2 made for learning the
filterbank, now measured.

**The threshold transfers much better.** At the dev-calibrated threshold of
0.9985, eval accuracy is 92.75%, with 7.97% of spoofs passed and 0.99% of real
clips flagged. The log-Mel CNN passed 32.51% of spoofs (Finding 4). Still,
eval's own EER point sits at 0.7333, so the served threshold is not where eval
would put it.

## Finding 6 — In-the-Wild: both models collapse on real-world deepfakes

Jobs 60300909, 60301093 and 60301094, 21 September 2026. In-the-Wild (Müller
et al.) contains 31,779 clips of 54 public figures (19,963 real, 11,816 fake),
collected from the internet rather than generated in a lab. It is used **for
evaluation only**: no model here was trained, selected or calibrated on it.
No min t-DCF is reported, because that metric needs the ASVspoof organisers'
speaker-verification scores, which only exist for ASVspoof.

| Model | LA eval EER | In-the-Wild EER | At the served threshold |
| --- | --- | --- | --- |
| **AASIST, `best.pth` (epoch 42)** | 3.17% | **37.15%** | 50.87% accuracy; 72.79% of real clips flagged, 9.16% of fakes passed |
| AASIST, epoch 85 | 3.02% | 35.17% | 48.22% accuracy; 79.84% flagged, 4.38% passed |
| CNN log-Mel, `best.pth` (epoch 5) | 10.15% | **58.54%** | 35.22% accuracy; 97.47% flagged, 9.55% passed |

**AASIST goes from 3.17% to 37.15% EER, about twelve times worse.** That falls
in the 30–40% range the literature led us to expect: detectors trained on
ASVspoof2019 LA generalise poorly to modern, real-world fakes. This gap, not
the LA number, is the fairer answer to "would this work on a clip someone
uploads today?", and the honest answer is no.

**The CNN is worse than chance.** An EER above 50% means its scores rank the
two classes the wrong way round: on this data it rates real clips as *more*
spoof-like than fakes. At its served threshold it flags 97.47% of real clips.
Flipping its output would give 41.46%, but that would mean choosing the model's
direction by looking at the test set, so it is not a legitimate result.

**In practice, the product would call most real audio fake.** Both models'
dev-calibrated thresholds sit at 0.998 or above, and In-the-Wild's real clips
score above them. Finding 4's warning about thresholds calibrated on dev is
much stronger here: at AASIST's served threshold nearly three in four real
clips get flagged.

The epoch 85 row is there for comparison, not as a headline. It was chosen
because it had the best LA-eval min t-DCF, which means it was chosen by
looking at a test set. Its 2-point improvement on In-the-Wild is also within
what a single seed could produce by chance.

## What has not been measured

- **AASIST under DP.** The port trains under Opacus, but the private AASIST
  run has not been done, so the cost of privacy exists only for the CNN.
- **AASIST with BatchNorm, or over several seeds** — the two cheapest ways to
  find out how much of the 3.17%-vs-0.83% gap comes from the GroupNorm swap.
- **The rest of the comparison table** in `APPROACH.md`: LCNN-LSTM-sum,
  AASIST-L and an SSL front-end.
- **A DP sweep.** One ε is a point, not the cost-of-privacy curve.
- **Any tuned run.** Five epochs, one learning rate, one batch size, throughout.
- **Variance.** Every result is a single seed. None of the gaps here have error
  bars, and the differences between adjacent epochs may not survive a reseed.

## Reproducing

```bash
sbatch hpc/train.slurm --no-dp                      # log-Mel baseline
sbatch hpc/train.slurm --no-dp --frontend lfcc      # LFCC ablation
sbatch hpc/train.slurm                              # DP arm
sbatch hpc/evaluate.slurm                           # score best.pth on eval
sbatch hpc/sweep_epochs.slurm <run-dir>             # score every epoch
sbatch --time=12:00:00 hpc/train.slurm --arch aasist --no-dp   # AASIST
sbatch hpc/evaluate.slurm --arch aasist                        # score it on LA eval
sbatch hpc/evaluate.slurm --dataset itw --arch aasist          # and on In-the-Wild
cd ai_model && python summarise_results.py <run-dir>
```
