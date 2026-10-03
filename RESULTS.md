# SoundSentinal — experimental results

Every number produced against the current architecture, with the run that
produced it. `APPROACH.md` records *what we chose and why*; this file records
*what happened*. When the two disagree about a number, this file is newer.

**All runs: 20–21 September 2026, Monash M3 (project `df37`), trained on
ASVspoof2019 LA** — except the two AASIST variants of 24 September in Finding 7,
one with RawBoost augmentation and one with ASVspoof 5 added to training, and
the two SSL-AASIST runs of 24–25 September in Finding 8 (XLS-R 300M in front of
AASIST, 100 epochs on an H100, with and without RawBoost), and the SSL-AASIST
run of 26 September in Finding 9, trained on LA plus SpeechFake, and the
run of 27–28 September in Finding 10, which adds Common Voice as bona fide
speech, and the run of 28 September in Finding 11, which adds Common Voice and
VoxPopuli. CNN runs are 5 epochs, batch 64, Adam at 1e-3,
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
| **Our SSL-AASIST + RawBoost, `best.pth` (epoch 91)** | XLS-R 300M | non-private | **0.79%** | **0.0143** |
| Our SSL-AASIST + RawBoost, LA + SpeechFake + Common Voice bona fide (epoch 4)† | XLS-R 300M | non-private | 1.39% | 0.0415 |
| Our SSL-AASIST + RawBoost, LA + SpeechFake + CV + VoxPopuli bona fide (epoch 4)† | XLS-R 300M | non-private | 2.76% | 0.0718 |
| Our SSL-AASIST + RawBoost, LA + SpeechFake (epoch 4)† | XLS-R 300M | non-private | 2.12% | 0.0649 |
| **Our AASIST + RawBoost, `best.pth` (epoch 51)** | raw waveform | non-private | **1.74%** | **0.0531** |
| **Our AASIST, `best.pth` (epoch 42)** | raw waveform | non-private | **3.17%** | **0.0909** |
| Our AASIST, LA + ASVspoof 5 train (epoch 12) | raw waveform | non-private | 5.47% | 0.1425 |
| Our SSL-AASIST, `best.pth` (epoch 75) | XLS-R 300M | non-private | 6.06% | 0.0828 |
| LFCC-GMM (official B2) | LFCC | non-private | 8.09% | 0.2116 |
| **Our CNN, epoch 2** | log-Mel | non-private | **9.60%** | **0.2124** |
| CQCC-GMM (official B1) | CQCC | non-private | 9.57% | 0.2366 |
| Our CNN, `best.pth` | log-Mel | non-private | 10.15% | 0.2350 |
| Our CNN, epoch 5 | LFCC | non-private | 13.10% | 0.2503 |
| Our CNN, `best.pth` | LFCC | non-private | 13.72% | 0.2463 |
| **Our CNN, epoch 3** | log-Mel | **DP, ε=0.48** | **17.57%** | **0.2609** |
| Our CNN, `best.pth` | log-Mel | DP, ε=0.48 | 17.80% | 0.2696 |

† Not a clean held-out number: SpeechFake trains on VCTK, the corpus LA's
bona fide speech comes from (Finding 9). Their point is In-the-Wild: **2.65%**,
2.71% and 3.55% respectively (Findings 9, 10 and 11).

Our AASIST's best single epoch on eval reaches 2.98% EER (epochs 59, 64) and
0.0807 min t-DCF (epochs 85, 95), but those are picked by looking at eval, so
the `best.pth` row is the one to quote. See Finding 5.

### The served system, and where it stands (30 September 2026)

SSL-AASIST + RawBoost trained on LA + SpeechFake (Finding 9), threshold set so
1% of held-out People's Speech is flagged (Finding 12). The model was never
trained, selected or calibrated on any test set below.

| Test set | What it is | EER | Real flagged | Fakes missed |
| --- | --- | --- | --- | --- |
| In-the-Wild | real-world recordings and deepfakes of public figures | 2.65% | **0.62%** | **3.61%** |
| SpeechFake test (English) | clean modern TTS/VC, 26 systems seen in training | 6.26% | 0.01% | **32.72%** |
| ASVspoof2019 LA eval | clean studio speech, 2019-era fakes | 2.12% | 0.00% | **46.25%** |

**On real-world audio the detector is strong; on clean synthetic speech it
misses a third to a half of fakes.** The EER column shows it can still rank
clean fakes well; the misses come from one threshold having to serve two kinds
of audio. Clean audio, real and fake, sits about 14 log-odds lower on its
scale than noisy real-world speech (Finding 13), and the threshold is set for
the latter.

What was tried and did not close that gap: adding real-world speech to
training (Findings 10, 11: the model learned each source's recording
conditions), and degrading every input through a codec, phone band or noise
(Finding 14: it blurred the fakes' artefacts instead of lifting clean audio).
What has not been tried: a separate threshold for clean audio, and scoring
the whole clip rather than its first 4 seconds.

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

## Finding 7 — RawBoost helps LA and hurts In-the-Wild; extra data does neither

Jobs 60380832 (AASIST + RawBoost algo 5, 100 epochs, 11h56m on an L40S) and
60383146 (AASIST on LA train + ASVspoof 5 train, 12 epochs, 5h22m on an A100),
24 September 2026. Scored by jobs 60386933/60386934 and 60384452/60386932.
Both are aimed at Finding 6, and each changes one thing against the AASIST
baseline: RawBoost distorts every training clip (convolutive + impulsive
noise, re-drawn each read); `--extra-train asvspoof5` adds 182,357 clips from
~2,000 crowdsourced speakers and newer TTS/VC attacks (ODC-By, same terms as
LA). Selection and calibration stay on LA dev for both, In-the-Wild is still
evaluation only, and neither run uses the other's change.

| Model | LA eval EER | min t-DCF | In-the-Wild EER | ITW at served threshold |
| --- | --- | --- | --- | --- |
| AASIST baseline (epoch 42) | 3.17% | 0.0909 | 37.15% | 72.79% of real flagged, 9.16% of fakes passed |
| **+ RawBoost** (epoch 51) | **1.74%** | **0.0531** | **48.78%** | 84.86% flagged, 10.96% passed |
| + ASVspoof 5 train (epoch 12) | 5.47% | 0.1425 | 38.14% | 29.43% flagged, 52.04% passed |

**RawBoost is the best LA model we have, and the worst on real-world audio.**
LA eval EER falls from 3.17% to 1.74% and min t-DCF from 0.0909 to 0.0531,
about 1.9× off the paper instead of 3.3×. The hardest attack improves most,
A18 from 10.48% to 4.25%; A17 barely moves (3.90% → 3.42%). On In-the-Wild, the same checkpoint
scores 48.78%, i.e. its scores barely separate real from fake. This is not
an evaluation bug: the scoring code is the one that produced 37.15% for the
baseline and 38.14% for the other run on the same day, and the LA number
comes from the same `best.pth`.

**Adding ASVspoof 5 changed where the threshold falls, not how well the model
ranks clips.** In-the-Wild EER is 38.14% against 37.15%, within what one seed
could produce. At the served threshold it flags 29% of real clips instead of
73%, which looks like progress, but it now passes 52% of fakes instead of 9%:
the operating point moved along roughly the same curve. It is also worse on
LA eval (5.47%, with A18 at 19.25%), which is expected when LA is an eighth of
the training data. It trained for 12 epochs, chosen to match the baseline's
number of optimiser steps rather than its number of passes, so it has seen
each LA clip 12 times rather than 100.

**What this says.** LA eval and In-the-Wild measure different things. LA eval
tests unseen *attacks* recorded under the same conditions as training;
RawBoost's channel distortions make the model better at exactly that. In-the-Wild
tests unseen *speakers, channels and generators at once*, and neither change
moved it. Two more corpora of lab-generated attacks on read speech do not
look like the missing ingredient. The approach the literature credits with
large In-the-Wild gains is a self-supervised front-end (wav2vec 2.0 / XLS-R),
already on the list below, and that is now the stronger candidate than more
augmentation or more ASVspoof-style data.

Caveats: one seed each; `best.pth` still chosen by LA dev (Finding 1), and no
per-epoch sweep has been run on either model; the two changes were not tried
together.

## Finding 8 — a pretrained front-end fixes In-the-Wild, but only with RawBoost

Jobs 60405665 (SSL-AASIST) and 60405666 (SSL-AASIST + RawBoost algo 5),
24–25 September 2026: 100 epochs each on an H100, 13h22m and 12h51m. Scored
by jobs 60434973–60434976 on L40S GPUs. SSL-AASIST puts XLS-R 300M, a
wav2vec 2.0 model pretrained on 436k hours of unlabelled speech in 128
languages, in front of AASIST's graph back-end and fine-tunes the whole
316M-parameter network (`ai_model/ssl_aasist.py`). Recipe: batch 14, Adam at a
constant 1e-6, weight decay 1e-4 — upstream's. Trained on LA alone, selected
and calibrated on LA dev, exactly like every row above.

| Model | LA eval EER | min t-DCF | In-the-Wild EER | ITW at served threshold |
| --- | --- | --- | --- | --- |
| AASIST baseline (epoch 42) | 3.17% | 0.0909 | 37.15% | 72.79% of real flagged, 9.16% of fakes passed |
| AASIST + RawBoost (epoch 51) | 1.74% | 0.0531 | 48.78% | 84.86% flagged, 10.96% passed |
| SSL-AASIST (epoch 75) | 6.06% | 0.0828 | 37.86% | 82.91% flagged, 0.30% passed |
| **SSL-AASIST + RawBoost** (epoch 91) | **0.79%** | **0.0143** | **11.21%** | **6.80% flagged, 18.56% passed** |

**The combination is the first model that works on real-world audio.**
In-the-Wild EER falls from 37.15% (the model the app serves) to 11.21%, and at
its served threshold it flags 6.8% of genuine clips instead of 73%. On LA eval
it is the best row in this file, below even the published AASIST row (0.79% /
0.0143 against 0.83% / 0.0275) — though that is a much smaller model without a
pretrained front-end, so it is not a like-for-like comparison. Its worst LA attack is A18 at 7.04%; the
spread across attacks is 0.00%–7.04%, against 0.02%–29.94% without RawBoost.

**Neither ingredient works alone.** The pretrained front-end without RawBoost
scores 37.86% on In-the-Wild — no better than plain AASIST — and 6.06% on LA,
worse than plain AASIST, with A15 at 29.94%. RawBoost without the pretrained
front-end made In-the-Wild *worse* (48.78%, Finding 7). Together they cut it by
a factor of three. A plausible reading: XLS-R already represents real-world
speech well, and RawBoost stops fine-tuning from over-writing that with "clean
VCTK channel = real"; without the pretrained features there is nothing
general for the augmentation to protect. That is an interpretation, not
something these four runs isolate.

**Published context.** The same architecture trained on LA is reported at
13.58% on In-the-Wild in one 2025 study (arXiv 2508.10559), so 11.21% is in
line with the literature rather than an outlier.

**A scoring bug found and ruled out.** These models are confident enough that
float32 softmax saturates: 15,307 In-the-Wild clips scored exactly P(spoof) =
1.0 and the plain run's EER was taken at "threshold 1.0000". An EER computed
inside a tie that size is arbitrary, so `evaluate.py` now scores on log-odds
(logit[spoof] − logit[real]), which ranks identically but never saturates
(commit d2a994c). All four runs were re-scored: every number moved by at most
0.04 points (37.82% → 37.86%), so the saturation was real but harmless here.
Every number in this file from now on is on the log-odds scale.

**What is still wrong.** At its served threshold the RawBoost model passes
18.56% of In-the-Wild fakes, and on LA eval 7.76% — the dev-calibrated
threshold (0.0142) still does not transfer (Finding 4). And selection is
weaker than ever: both runs reach 0.00% on LA dev, after which every epoch
ties, so `best.pth` (epochs 75 and 91) is not a meaningful choice among the
last ~50 epochs (Finding 1). SpeechFake's dev split fixes this for the next
run.

Caveats: one seed each; no per-epoch sweep (the plain run's per-epoch files
for epochs 1–10 were deleted on 25 September to stay inside the project quota;
every other epoch of both runs is kept); XLS-R's pretraining corpus is not
public in full, so overlap with In-the-Wild's speakers cannot be ruled out,
though nothing here was trained or selected on In-the-Wild.

**Next:** the same model retrained with SpeechFake added — Finding 9.

## Finding 9 — SpeechFake gets In-the-Wild to 2.65%, but the threshold does not transfer

Job 60452188, 26 September 2026: SSL-AASIST + RawBoost algo 5, trained on LA
train plus SpeechFake's bilingual training split (`--extra-train speechfake`,
704,862 clips from 30 open-source TTS, voice-conversion and vocoder systems,
English and Chinese, CC BY 4.0). 4 epochs — about 1.15× the optimiser steps
of Finding 8's 100 LA epochs — 11h43m on an H100 at 5.28 steps/s. Same recipe
otherwise (batch 14, constant lr 1e-6). Scored by jobs 60452189 (LA eval) and
60452190 (In-the-Wild) on L40S GPUs; JSON in
`~/df37_scratch/mkha0155/checkpoints/ssl-aasist/rawboost5/plus-speechfake/nodp/`.

Selection and calibration used **LA dev + SpeechFake dev** (142,307 clips),
because LA dev had saturated at 0.00% for SSL-AASIST (Finding 8). Unlike LA
dev, it still separated epochs:

| Epoch | Train acc | Dev EER | Dev threshold |
| --- | --- | --- | --- |
| 1 | 86.32% | 6.09% | 0.0142 |
| 2 | 96.58% | 4.28% | 0.0090 |
| 3 | 97.51% | 4.05% | 0.0039 |
| **4** (`best.pth`) | 98.10% | **3.01%** | 0.0026 |

| Model | LA eval EER | min t-DCF | In-the-Wild EER | ITW at served threshold |
| --- | --- | --- | --- | --- |
| AASIST baseline (epoch 42) | 3.17% | 0.0909 | 37.15% | 72.79% of real flagged, 9.16% of fakes passed |
| SSL-AASIST + RawBoost, LA only (epoch 91) | 0.79% | 0.0143 | 11.21% | 6.80% flagged, 18.56% passed |
| **+ SpeechFake** (epoch 4) | 2.12%† | 0.0649† | **2.65%** | **46.02% flagged, 0.14% passed** |

† See the note under the summary table: SpeechFake contains VCTK.

**In-the-Wild EER is 2.65%, under the 5% target** and a factor of 4.2 below
Finding 8. At its own EER threshold the model gets 19,434 of 19,963 real clips
and 11,503 of 11,816 fakes right. This matches the 2.63% reported for training
on SpeechFake's bilingual subset in arXiv 2606.08038, although that study's
setup was not identical to ours. It supports that study's conclusion: what
LA-only training lacked was **generator diversity** — 30 systems against LA's
six — and not hours of audio (ASVspoof 5 added 182k clips and did nothing,
Finding 7).

**But the served threshold makes the model unusable on real-world audio.**
EER measures ranking — whether fakes score above real clips — and the ranking
is now very good. A deployed detector also needs a threshold, and the one
calibrated on dev (P(spoof) = 0.0026) flags **46% of genuine In-the-Wild clips
as fake**. The EER threshold on In-the-Wild sits at P ≈ 0.999, nearly three
orders of magnitude higher on the odds scale. Real-world speech gets far higher
"fake" scores than the clean read speech that dev's bona fide consists of
(VCTK, LibriTTS, AISHELL), while still scoring below real-world fakes. The
ranking transferred; the scale did not. It is Finding 4 again, much larger.

The threshold must **not** be fixed by calibrating on In-the-Wild: that turns
the test set into a calibration set, and 2.65% would stop measuring anything.
The fix is a held-out set of noisy, real-world *bona fide* speech that is not
In-the-Wild, to calibrate on. This is the same gap as before — no noisy real
speech anywhere in training or dev — showing up in calibration instead of in
ranking.

**On LA eval it got worse** (0.79% → 2.12%; worst attack A10 at 6.63%, best
A13 at 0.10%). LA is now 3.5% of the training data, so this is expected; and
because of the VCTK overlap the LA number no longer measures generalisation
for this model anyway.

### Follow-up: recalibrating the threshold on Common Voice (27 September)

Jobs 60472073 (calibrate) and 60472074/60472075 (score). `ai_model/calibrate.py`
set the threshold so that 5% of genuine **Common Voice** English clips are
flagged — volunteers reading on their own microphones (CC0, never seen in
training). The 5% target was fixed before In-the-Wild was scored at it, and
In-the-Wild was scored once. It wrote `best_calibrated.pth`; `best.pth` keeps
the dev threshold.

| Threshold | Common Voice real flagged (held-out half) | ITW real flagged | ITW fakes passed | LA eval fakes passed |
| --- | --- | --- | --- | --- |
| Dev EER, P = 0.0026 | 33.26% | 46.02% | 0.14% | 4.08% |
| **Common Voice 5%, P = 0.7457** | **5.52%** | **15.64%** | **1.17%** | 18.09% |

**It worked on Common Voice and only partly on In-the-Wild.** The fitted rate
held on the held-out half (5.52% against a 5.00% fit), and In-the-Wild's false
flags fell from 46% to 16% while fakes passed rose from 0.14% to 1.17%. But
16% is three times the target: In-the-Wild's genuine clips — interviews,
speeches, broadcast audio — score higher still than people reading at home.
Common Voice is closer to In-the-Wild than dev's studio speech, not close
enough. The EER (2.65%) is unchanged by construction: moving a threshold never
changes the ranking.

**LA eval now passes 18% of fakes.** Clean studio spoofs score low on this
model's scale, below a threshold set for noisy real speech. One threshold
cannot suit both clean lab audio and real-world audio; for a tool whose users
upload real-world clips, the real-world one is the right choice, but the
trade-off should be stated.

**Second attempt: VoxPopuli (jobs 60478739–60478741).** Chosen *because*
Common Voice fell short on In-the-Wild — so In-the-Wild informed that choice,
and both attempts are reported here. European Parliament speeches (CC0, none
of In-the-Wild's speakers), 9,000 clips split by speaker (719 speakers), same
5% target:

| Threshold fitted on | Held-out real flagged | ITW real flagged | ITW fakes passed | LA eval fakes passed |
| --- | --- | --- | --- | --- |
| Dev EER (clean read speech) | 35.19% (VoxPopuli) / 33.26% (CV) | 46.02% | 0.14% | 4.08% |
| Common Voice, P = 0.7457 | 5.52% | **15.64%** | 1.17% | 18.09% |
| VoxPopuli, P = 0.5308 | 3.62% | 19.26% | 0.98% | 15.78% |

**It did worse.** The model scores parliamentary speech as *more* genuine
than people reading at home, so the threshold came out lower and flagged more
of In-the-Wild. One likely reason is the caveat in `voxpopuli.py`: XLS-R was
pretrained on unlabelled VoxPopuli, so this audio is familiar to its
features. Either way, In-the-Wild's genuine clips score higher than any real
speech we can calibrate on, so **no threshold chosen without looking at
In-the-Wild gets its false flags near 5%**. Calibration stops here, as
committed before this attempt: a third set chosen after seeing two
In-the-Wild results would be tuning on the test set by increments.

**Conclusion.** The fault is in the scores, not the threshold. Genuine
real-world recordings look suspicious to this model because nothing real and
noisy was in its training data. The fix is to train with real-world bona fide
speech — e.g. VoxPopuli, Common Voice or People's Speech (all ungated,
commercially usable) — so the model learns that noisy, broadcast and
home-recorded real speech is real.

Caveats: one seed; four epochs, with dev EER still falling at epoch 4 (so
more epochs might help); `best.pth` is the last epoch, so selection did not
have to choose; XLS-R's pretraining data may overlap In-the-Wild's speakers
(Finding 8). Nothing was trained, selected or calibrated on In-the-Wild.

## Finding 10 — training on real-world bona fide speech makes the threshold worse

The pre-registration below is unchanged from 27 September. The result follows
it, under "Result (28 September)".

Written 27 September 2026, **before** the run is submitted. Nothing below may
be changed after In-the-Wild is scored; the result is reported whatever it is.

**Question.** Does adding noisy, home-recorded real speech to training, labelled
bona fide, stop the model scoring genuine real-world recordings as suspicious
(Finding 9), without losing its ranking?

**The one change.** Finding 9's run plus `--extra-bonafide commonvoice`: Common
Voice English's **train** split, 33,614 clips (`commonvoice.py`). Everything
else is held fixed: SSL-AASIST, RawBoost 5, `--extra-train speechfake`, 4
epochs, batch 14, constant lr 1e-6, class weights recomputed from the data
(bona fide goes from ~78k to ~112k of ~739k clips). Selection is unchanged —
LA dev + SpeechFake dev, `best.pth` by dev EER — so Common Voice influences
only the weights, not which epoch is chosen.

**Calibration**, fixed now: `calibrate.py --commonvoice-split test`, Common
Voice English's **test** split (16,386 clips; Common Voice puts each speaker
in one split only), n = 10,000, seed 42, target 5% of genuine clips flagged.
`calibrate.py` refuses the train split for this checkpoint.

**Control.** The Finding 9 model is recalibrated the same way (test split,
same n and seed) and scored on In-the-Wild at that threshold, so both models
are compared at thresholds chosen by the identical procedure. Its earlier
15.64% sampled both Common Voice splits.

**What is reported**, in this order:

1. In-the-Wild EER (ranking). Must stay **under 5%**.
2. **In-the-Wild real clips flagged** at the Common Voice test threshold — the
   primary number. Compared with the control.
3. In-the-Wild fakes passed at that threshold.
4. VoxPopuli real clips flagged at that threshold (`calibrate.py
   --calibration-set voxpopuli` on the calibrated checkpoint, whose "old
   threshold" line is then the Common Voice one): a second, unseen kind of
   real speech.
5. LA eval EER / min t-DCF, with Finding 9's VCTK caveat.

**Reading the result**, decided in advance:

- **Fixed:** ITW EER < 5% and ≤ 5% of ITW real clips flagged. Serve it.
- **Improved:** ITW EER < 5% and real flagged clearly below the control (≥ 5
  points), but above 5%. Report it; decide on serving with the numbers.
- **No effect:** real flagged within 5 points of the control. The problem is
  not the absence of noisy bona fide in training.
- **Broken ranking:** ITW EER ≥ 5%. The added data cost more than it bought.

In-the-Wild is scored once for this model. If the outcome is not "fixed", the
next step is written up and argued for, not run as a quick variant — a second
data mix chosen after seeing this result would be tuning on In-the-Wild.

### Result (28 September)

Job 60493491: 12h20m on an H100, no errors. Class weights came out
`[3.413, 0.586]` (111,902 bona fide, 651,954 spoof). Scored by jobs 60500934
(calibration), 60500935 (In-the-Wild), 60500936 (VoxPopuli) and 60500939 (LA
eval). The control used jobs 60493492 (calibration), 60500937 (In-the-Wild) and
60500938 (VoxPopuli). All ran on L40S GPUs at commit `677066e`. The JSON is in
`.../plus-speechfake/plus-commonvoice-bonafide/nodp/` and `.../plus-speechfake/nodp/`.

| Epoch | Train acc | Dev EER | Dev threshold |
| --- | --- | --- | --- |
| 1 | 89.25% | 5.20% | 0.0171 |
| 2 | 97.13% | 3.12% | 0.0407 |
| 3 | 97.90% | 3.20% | 0.0076 |
| **4** (`best.pth`) | 98.34% | **2.48%** | 0.0045 |

The five numbers, in the pre-registered order. Both models use a threshold
fitted on the Common Voice **test** split (n 10,000, seed 42, target 5%):

| | Control (Finding 9 model) | + Common Voice bona fide |
| --- | --- | --- |
| Common Voice test threshold | P = 0.9766 (log-odds +3.73) | P = 0.000509 (log-odds −7.58) |
| Common Voice real flagged, held-out half | 5.50% | 6.70% |
| 1. In-the-Wild EER | 2.65% | **2.71%** |
| 2. **In-the-Wild real flagged** | **8.04%** | **53.69%** |
| 3. In-the-Wild fakes passed | 1.74% | 0.12% |
| 4. VoxPopuli real flagged (9,000 clips) | 0.93% | 52.41% |
| 5. LA eval EER / min t-DCF | 2.12% / 0.0649† | 1.39% / 0.0415† |

† SpeechFake contains VCTK, so neither LA row is a clean held-out number
(Finding 9). The new model's worst attack is A11 (4.23%) and its best is A13
(0.00%).

**The result is none of the four pre-registered outcomes.** The ranking held:
the In-the-Wild EER stayed at 2.71%, under 5%, so this is not "broken ranking".
But real clips flagged went from 8.04% to 53.69%, **45.6 points worse** than
the control. "Fixed" and "improved" need it lower. "No effect" needs it within
5 points. The categories did not allow for the change making things worse, and
the result is reported as that rather than filed under the nearest heading.

**Why: training on Common Voice made Common Voice useless for calibration.**
The new model is extremely confident that Common Voice audio is real. Its
held-out Common Voice scores span log-odds −7.79 to −7.44 from the 5th to the
99th percentile, a band 0.35 wide; the control's same percentiles span −7.10 to
+8.30. Flagging "5% of Common Voice" therefore put the threshold inside that
narrow band, at −7.58. Nearly every real recording that does not sound like
Common Voice scores above it, so about half of In-the-Wild and of VoxPopuli
gets flagged. Common Voice keeps each speaker in one split, so its test split
has new speakers. It does not have new recording conditions: same website, same
kind of microphones, same reading task. The model learned those conditions, and
held-out speakers did not hold them out.

**This is a flaw in the pre-registration, not in the run.** Calibration must
use a source of real speech the model was never trained on. Choosing the other
split of the training corpus met the letter of that and not the point of it.

**What the run does show:**

- **Adding real speech did not hurt the ranking.** In-the-Wild EER moved from
  2.65% to 2.71%, and LA eval improved from 2.12% / 0.0649 to 1.39% / 0.0415.
  The data is safe to train on. The question is only how to set the threshold
  afterwards.
- **It did not make real-world speech look real in general.** It made Common
  Voice look real. On the new model, Common Voice's 95th-percentile score is
  log-odds −7.57, and VoxPopuli's is −0.25. A quarter of VoxPopuli scores
  above −5.29, which is far above anything Common Voice produces.
- **The control's threshold is unstable.** Two samples from the same corpus
  gave thresholds P = 0.7457 (both splits, Finding 9) and P = 0.9766 (test
  split only). Their In-the-Wild false flags were 15.64% and 8.04%. A threshold
  that moves by 2.6 on the log-odds scale when the sample changes is not a
  property of the model that can be relied on.

**On serving the control at 8.04%.** Of the thresholds fitted for the
Finding 9 model, this one gives the lowest real-world false flags: 8.04%, with
1.74% of fakes passed. Its procedure was fixed before In-the-Wild was scored.
But it is the third threshold for that model scored on In-the-Wild
(15.64%, 19.26%, 8.04%). Choosing it *because* it scored best is choosing on
In-the-Wild. If it is served, the reason has to be one that does not depend on
those three numbers, and `/result` should state the measured rates.

**The next step, argued rather than run.** The mistake to avoid is
calibrating on a source that is also in training. The design that avoids it
holds out whole *sources*, not speakers:

1. Train with bona fide speech from two or more real-world sources, e.g.
   Common Voice and People's Speech.
2. Calibrate on a third source that stays out of training entirely. VoxPopuli
   is a candidate, with the caveat that XLS-R was pretrained on it (Finding 9).
3. Fix all three choices, the 5% target and the outcome categories before
   submitting, and score In-the-Wild once.

This is a new data mix chosen after seeing this result, which the
pre-registration warned about. What keeps it from being tuning is that it is
motivated by a mechanism found in the calibration data (the collapsed score
band), not by In-the-Wild's numbers. That should be checked when it is
pre-registered. Nothing was trained, selected or calibrated on In-the-Wild in
this finding.

## Finding 11 — two real-speech sources in training, a third held out to calibrate: no effect

The pre-registration below is unchanged from 28 September. The result follows
it, under "Result (29 September)".

Written 28 September 2026, **before** anything is submitted. Nothing below may
be changed after In-the-Wild is scored. The result is reported whatever it is.

**Question.** If the model trains on real speech from more than one recording
setup, and the threshold is set on a source it never trained on, does it stop
flagging genuine real-world recordings, without losing its ranking?

**Why this design, and why it is not tuning on In-the-Wild.** Finding 10 failed
for a reason visible without In-the-Wild: the model's scores on Common Voice
test collapsed into a band 0.35 log-odds wide, because it had trained on Common
Voice's recording conditions. This design fixes that specific mechanism: the
threshold comes from a source held out of training, and training spreads
"real" across two setups. The choice of sources was made from their licences
and from what XLS-R was pretrained on, not from any In-the-Wild number.

**The change.** Finding 9's run plus `--extra-bonafide commonvoice+voxpopuli`:

- Common Voice English, **train** split (33,614 clips): people reading at home
  on their own microphones. CC0.
- VoxPopuli English, **train** shards 00000-00005 (~36k clips): European
  Parliament speeches through the chamber's broadcast microphones. CC0.

Everything else is held fixed: SSL-AASIST, RawBoost 5, `--extra-train
speechfake`, 4 epochs, batch 14, constant lr 1e-6, class weights recomputed
from the data. Selection is unchanged: LA dev + SpeechFake dev, `best.pth` by
dev EER.

**Calibration, fixed now.** `calibrate.py --calibration-set peoples_speech`:
People's Speech `clean` **test** split, archive.org talks, lectures, meetings
and proceedings. It is CC-BY, so commercial use is allowed. It is in neither
training nor XLS-R's pretraining data. n = 10,000, seed 42, target 5% of
genuine clips flagged, halves split by source recording. Before sampling,
every recording whose name matches one of In-the-Wild's 54 speakers is
dropped. The patterns are in `peoples_speech.py` and were fixed before any
download. `calibrate.py` refuses to set this checkpoint's threshold on Common
Voice or VoxPopuli.

**Control.** The Finding 9 model, calibrated on People's Speech the same way
(same n, seed and target) and scored on In-the-Wild at that threshold. Both
models then have thresholds chosen by the identical procedure, from the same
unseen source. This is the **fourth** threshold for the Finding 9 model to be
scored on In-the-Wild. It is here for attribution only. It is not a serving
candidate, whatever it scores.

The Finding 10 model is **not** re-scored: Finding 10 committed to scoring it
on In-the-Wild once.

**What is reported**, in this order:

1. In-the-Wild EER. Must stay **under 5%**.
2. **In-the-Wild real clips flagged** at the People's Speech threshold. This
   is the primary number, compared with the control.
3. In-the-Wild fakes passed at that threshold.
4. People's Speech real clips flagged, check half. This shows whether the
   threshold holds on recordings it was not fitted to.
5. Diagnostics at that threshold, using `calibrate.py --measure-only`; they
   choose nothing. Common Voice test (n 10,000, seed 42) and VoxPopuli
   held-out speakers (validation and test clips whose speakers are in no
   fetched train shard, all of them up to 10,000), for both models.
6. LA eval EER / min t-DCF, with Finding 9's VCTK caveat.

**Reading the result**, decided in advance. "Control" means the control's
In-the-Wild real flagged rate. The first line that applies wins:

- **Broken ranking:** new model's ITW EER ≥ 5%. The added data cost more than
  it bought.
- **Fixed:** ≤ 5% of ITW real clips flagged. Serve the new model at the People's
  Speech threshold, and state the measured rates on `/result`. If the control
  is also ≤ 5%, the fix came from the held-out calibration source rather than
  from training, and the writeup says so.
- **Improved:** ≥ 5 points below the control, but above 5%. Report it and decide
  on serving with the numbers.
- **No effect:** within 5 points of the control, either way.
- **Worse:** ≥ 5 points above the control. Training on real speech still teaches
  the model its sources' recording conditions rather than what real speech is.

Finding 10's categories left no room for "worse", and it happened. This time
there is a category for it.

In-the-Wild is scored once for each of the two models. If the outcome is not
"fixed", the next step is written up and argued for, not run as a quick
variant.

### Result (29 September)

Job 60504169: 12h49m on an H100, no errors, at commit `2ddcdba`. Class weights
came out `[2.697, 0.614]` (148,400 bona fide, 651,954 spoof). Scored by jobs
60504170 (People's Speech calibration), 60504171 (In-the-Wild), 60504172 and
60504174 (Common Voice and VoxPopuli, measure-only) and 60504176 (LA eval). The
control used jobs 60504178, 60504180, 60504182 and 60504184. All scoring ran on
L40S GPUs. The JSON is in
`.../plus-speechfake/plus-commonvoice+voxpopuli-bonafide/nodp/` and
`.../plus-speechfake/nodp/`.

| Epoch | Train acc | Dev EER | Dev threshold |
| --- | --- | --- | --- |
| 1 | 91.54% | 5.25% | 0.0259 |
| 2 | 97.46% | 4.30% | 0.0101 |
| 3 | 98.17% | 3.71% | 0.0069 |
| **4** (`best.pth`) | 98.63% | **2.96%** | 0.0064 |

The numbers, in the pre-registered order. Both models use a threshold fitted
on People's Speech `clean` test (n 10,000, seed 42, target 5%; 5,328 clips to
fit, 4,672 to check, halves split by speaker):

| | Control (Finding 9 model) | + CV + VoxPopuli bona fide |
| --- | --- | --- |
| People's Speech threshold | P = 0.9748 (log-odds +3.66) | P = 0.7454 (log-odds +1.07) |
| 1. In-the-Wild EER | 2.65% | **3.55%** |
| 2. **In-the-Wild real flagged** | **8.21%** | **8.90%** |
| 3. In-the-Wild fakes passed | 1.70% | 2.29% |
| 4. People's Speech real flagged, check half | 3.72% | 6.93% |
| 5a. Common Voice test real flagged (10,000 clips) | 5.30% | 0.08% |
| 5b. VoxPopuli held-out speakers real flagged (1,794 clips) | 0.56% | 0.00% |
| 6. LA eval EER / min t-DCF | 2.12% / 0.0649† | 2.76% / 0.0718† |

† SpeechFake contains VCTK, so neither LA row is a clean held-out number
(Finding 9). The new model's worst attack is A11 (7.34%) and its best is A13
(0.00%).

**The outcome is "no effect".** The ranking held (3.55%, under 5%), so it is
not "broken ranking". Real In-the-Wild clips flagged went from 8.21% to 8.90%,
0.69 points worse: within 5 points of the control, so neither "fixed",
"improved" nor "worse".

**What the run shows:**

- **Holding the calibration source out of training fixed Finding 10's
  failure.** The threshold no longer lands in a collapsed band, and the new
  model flags 8.90% of real In-the-Wild clips rather than 53.69%. But the
  control, calibrated the same way, gets 8.21%, so the fix came from the
  calibration source, not from the training data.
- **The model learned its two real-speech sources, not real speech.** It
  flags 0.08% of Common Voice and 0.00% of VoxPopuli, both in training, and
  8.90% of In-the-Wild, which is not. It is also less stable on the held-out
  source: 6.93% of People's Speech's check half is flagged against a 5%
  target, where the control misses by 1.3 points the other way. Half of People's
  Speech sits in a narrow band (5th–25th percentile log-odds −8.21 to −8.19),
  the same pattern Finding 10 showed on Common Voice, but only for part of the
  corpus.
- **It cost ranking.** In-the-Wild EER rose from 2.65% to 3.55%, and LA eval
  from 2.12% / 0.0649 to 2.76% / 0.0718. Finding 10's Common Voice-only run
  held both, so adding VoxPopuli is the likely cause, though one seed cannot
  separate that from noise.
- **The control's threshold is now reproducible across sources.** Common
  Voice test (Finding 10) and People's Speech gave log-odds +3.73 and +3.66,
  and In-the-Wild real flagged of 8.04% and 8.21%. Two independent held-out
  corpora agreeing is evidence that the Finding 9 model's threshold is a
  property of the model, which Finding 10 had doubted after the both-splits
  sample gave +1.07.

**On serving.** The pre-registration says the control is not a serving
candidate here, whatever it scores, and that stands: this is the fourth
threshold for the Finding 9 model scored on In-the-Wild. Any case for serving
it has to rest on something other than those scores. The strongest such case is
the agreement above, measured without In-the-Wild: two held-out sources, the
same procedure, the same threshold to within 0.07 log-odds.

**The next step, argued rather than run.** Two findings in a row show that
adding real speech to training teaches the model those recordings' conditions
and does not lower real-world false flags. Another data mix would be the
third attempt at the same idea. The options are:

1. **Serve the Finding 9 model at the People's Speech threshold** (P = 0.9748),
   with `/result` stating the measured rates: about 8% of real recordings
   flagged, under 2% of fakes passed. It needs to be decided on the
   cross-source agreement, and the writeup has to say that four thresholds
   were scored on In-the-Wild.
2. **Change the target, not the data.** A 5% target on clean-ish speech
   became about 8% on In-the-Wild for every threshold that transferred. A
   lower target trades fakes passed for fewer false flags; choosing it would
   need its own pre-registration, on held-out sources only.
3. **Report a band instead of a verdict.** The UI already draws a graduated
   scale; an "uncertain" zone around the threshold, sized from People's Speech,
   would fit the "instrument, not a verdict machine" thesis without another
   training run.

Nothing was trained, selected or calibrated on In-the-Wild in this finding.

## Finding 12 — a stricter target: 0.62% of real clips flagged, 3.61% of fakes passed

The pre-registration below is unchanged from 29 September. The result follows
it, under "Result (29 September)".

Written 29 September 2026, **before** anything is submitted. Nothing below may
be changed after In-the-Wild is scored. The result is reported whatever it is.

**Question.** The served model (Finding 9, People's Speech threshold) flags
8.21% of real In-the-Wild clips and passes 1.70% of fakes. Its ranking allows
about 2.65% of each (the EER). If the threshold is set so that 1% of People's
Speech is flagged instead of 5%, how much lower do real-world false flags go,
and what does it cost in fakes passed?

**What is fixed.** Everything from Finding 11's control except the target:
the Finding 9 checkpoint, People's Speech `clean` test, n 10,000, seed 42,
halves split by speaker, so the fit and check halves are the same clips as
before and only the quantile changes. `--target-frr 0.01`. Output
`best_calibrated_peoples_speech_1pct.pth` beside the 5% file, which is kept.

**Why 1%, and what it is not.** The target is a product decision made by the
project owner: fewer false accusations of real speakers, accepting more missed
fakes. It is a round number, not the result of a search. But the *direction*
was chosen after seeing that a 5% target became 8.21% on In-the-Wild, so this
choice is informed by an In-the-Wild number. That is disclosed here rather
than hidden. It is the fifth threshold for this model to be scored on
In-the-Wild, and the **last**: no further target is tried on this model.

**What is reported:**

1. People's Speech real flagged, check half (the target is 1%).
2. **In-the-Wild real flagged** at the new threshold. Primary number.
3. **In-the-Wild fakes passed** at the new threshold.
4. LA eval fakes passed at the new threshold, per attack, as a view of which
   attack types slip through.

The In-the-Wild EER does not change: same model, same scores.

**Expected** (written down so it can be wrong): real flagged 3–4%, fakes
passed around 3%.

**Decision, fixed now.** Serve the 1% threshold unless In-the-Wild fakes
passed exceeds **5%**, in which case keep serving the 5% threshold. Both
outcomes and the rule are recorded before the score exists. Whichever is
served, `/result` states its measured In-the-Wild rates.

### Result (29 September)

Jobs 60549794 (calibration, 1m46s), 60549795 (In-the-Wild) and 60549796 (LA
eval), all on L40S GPUs at commit `d1df9ce`. The JSON is in
`.../plus-speechfake/nodp/` (`best_calibrated_peoples_speech_1pct.json`,
`eval_itw_20260929-120334.json`, `eval_eval_20260929-120817.json`).

| | People's Speech 5% (Finding 11 control) | **People's Speech 1%** |
| --- | --- | --- |
| Threshold | P = 0.9748 (log-odds +3.66) | P = 0.999658 (log-odds +7.98) |
| 1. People's Speech real flagged, check half | 3.72% | **0.60%** |
| 2. **In-the-Wild real flagged** | 8.21% | **0.62%** (124 of 19,963) |
| 3. **In-the-Wild fakes passed** | 1.70% | **3.61%** (426 of 11,816) |
| In-the-Wild accuracy | 94.21% | 98.27% |
| 4. LA eval fakes passed | not scored | **46.25%** (29,543 of 63,882) |
| LA eval real flagged | not scored | 0.00% |

In-the-Wild EER is 2.65% and LA eval 2.12% / 0.0649, unchanged, as expected.

**The prediction was wrong in the good direction.** Real clips flagged fell
to 0.62%, not 3–4%. Fakes passed rose to 3.61%, close to the ~3% expected.
The threshold held on its own check half (0.60% against a 1% fit) and carried
over to In-the-Wild almost exactly, which the 5% threshold did not (3.72% on
the check half, 8.21% on In-the-Wild). Near the top of the score range, real
speech from both sources thins out the same way; in the middle, In-the-Wild's
noisier recordings sit higher than People's Speech's.

**Decision, by the pre-registered rule: serve the 1% threshold.** Fakes passed
is 3.61%, under the 5% limit.

**The cost: clean studio fakes.** At this threshold LA eval passes 46% of its
fakes. This is the same trade Finding 9's follow-up found at a lower threshold
(18% at P = 0.7457), taken further. LA's attacks are 2019-era TTS and voice
conversion rendered as clean studio audio, and this model scores them lower
than real-world fakes; a threshold set for noisy real speech sits above many of
them. In-the-Wild's fakes, which are what a user of this tool is likely to
upload, pass at 3.61%. `/result` has to say both: the rate is measured on
real-world audio, and clean synthetic speech in the style of older systems is
caught far less reliably.

**Not reported: LA fakes passed per attack.** `evaluate.py` saves per-attack
EERs, not per-clip scores, so the pass rate of each attack at a given
threshold cannot be recovered from the JSON. Reporting it needs `evaluate.py`
to write scores and one more LA run. The per-attack EERs (best A13 0.10%,
worst A10 6.63%) are unchanged from Finding 9.

This was the last threshold tried on this model.

## Finding 13 — clean modern fakes slip through too: 32.7% passed

The pre-registration below is unchanged. The result follows it, under
"Result (30 September)".

Written 30 September 2026, before the job is submitted. A diagnostic: it
chooses no threshold and trains nothing.

**Question.** Finding 12's served threshold passes 46% of LA eval's fakes,
which are clean studio audio from 2019-era systems. Is that about 2019-era
synthesis, or about clean audio? Modern commercial cloning also produces clean
audio, and if it behaves like LA the served model misses the fakes users worry
about most.

**What is scored.** The served checkpoint (`best_calibrated_peoples_speech_1pct.pth`)
on SpeechFake-BD's baseline **test** split, English rows only: 208,655 clips,
189,455 fakes from 26 TTS, voice-conversion and vocoder systems, 19,200 real
(LibriTTS and VCTK). `evaluate.py --dataset speechfake`, which now also reports,
for every attack, the share of its fakes that pass at the served threshold.

**Two limits, stated before the number exists.**

- **Seen generators.** The model trained on SpeechFake's train split, which
  holds the same 26 systems. So this is an optimistic test: a high pass rate is
  damning, a low one is an upper bound on how well it catches clean modern
  fakes, not evidence it catches unseen ones.
- **The real rows are not held out in kind.** LibriTTS and VCTK are in
  training, so the real-flagged rate here says little about users' recordings.

**Reading it, decided now** (pooled English fakes passed at the served
threshold):

- **≤ 10%:** the 46% is about 2019-era synthesis. Keep the threshold and the
  existing caveat on `/result`.
- **10–25%:** mixed. Report the per-system table and name the systems that
  pass most on `/result`.
- **> 25%:** clean modern synthesis slips through as well. The `/result`
  caveat is rewritten to say so plainly, and the next step is a model change
  (clean modern fakes in training), not another threshold: Finding 12 was the
  last threshold for this model.

### Result (30 September)

Job 60583939, 22m27s on an L40S at commit `da86162`. JSON:
`.../plus-speechfake/nodp/eval_speechfake-en_20260930-125040.json`.

| | SpeechFake-BD test, English |
| --- | --- |
| EER | 6.26% (at log-odds −5.98) |
| **Fakes passed at the served threshold (+7.98)** | **32.72%** (61,983 of 189,455) |
| Real flagged at the served threshold | 0.01% (1 of 19,200; real rows are from training corpora) |

**The outcome is "> 25%": clean modern synthesis slips through as well.**
Nearly a third of fakes pass, from systems the model trained on. Unseen
systems can only be expected to do worse.

**The ranking is mostly fine; the threshold is in the wrong place for clean
audio.** 20 of the 26 systems have an EER under 1.2%, yet many pass at the
served threshold. The best balance point on this data is log-odds −5.98, and
the served threshold is +7.98, 14 units higher. Clean audio of both kinds sits
low on this model's scale; the threshold was set on noisy real speech, which
sits high. The same split shows in Finding 12's LA row (46% passed at 2.12%
EER). A single threshold cannot serve both.

| System | EER | Passed at served threshold | n |
| --- | --- | --- | --- |
| BigVGAN | 56.07% | **100.00%** | 8,400 |
| MeloTTS | 3.91% | **93.42%** | 8,697 |
| ParlerTTS | 1.73% | **85.32%** | 5,417 |
| CosyVoice | 11.65% | **80.88%** | 19,364 |
| WaveGlow | 10.83% | **66.52%** | 3,638 |
| DiffGANTTS | 0.50% | 64.80% | 6,000 |
| GPTSoVITS | 3.40% | 47.15% | 6,057 |
| FastSpeech2 | 0.51% | 40.47% | 3,000 |
| FishSpeech | 1.19% | 39.10% | 8,757 |
| ChatTTS | 0.21% | 24.27% | 8,744 |
| OpenVoiceTTS | 0.28% | 19.83% | 10,167 |
| HifiGAN | 1.12% | 15.26% | 8,321 |
| StarGAN | 0.43% | 13.94% | 9,642 |
| Tortoise | 0.29% | 12.97% | 12,377 |
| FireRedTTS | 0.08% | 11.95% | 3,781 |
| FastDiff | 0.66% | 11.07% | 8,400 |
| StyleTTS2 | 0.19% | 10.27% | 5,715 |
| OpenVoice | 0.04% | 5.50% | 8,697 |
| PortaSpeech | 0.11% | 3.00% | 4,564 |
| SeedVC | 0.11% | 2.07% | 5,995 |
| ProDiffTTS | 0.15% | 2.03% | 9,200 |
| Tacotron2 | 0.03% | 0.58% | 6,000 |
| GlowTTS | 0.02% | 0.57% | 6,000 |
| ParallelWaveGAN | 0.10% | 0.45% | 5,943 |
| WaveNet | 0.08% | 0.42% | 2,400 |
| FullBandMelGAN | 0.04% | 0.36% | 4,179 |

BigVGAN is the one system the model cannot rank at all (56% EER): a
high-quality neural vocoder resynthesising real speech leaves little for it to
find. CosyVoice and WaveGlow are weak at ranking too. The rest pass because of
where the line is, not because the model cannot tell them apart.

**Done, as pre-registered:** the `/result` caveat now says plainly that clean
synthetic speech, old and modern, passes a third of the time or more, and
this set is a third row in its error-rate table
(`checkpoints/best.measured-speechfake.json`).

**The next step is a model change, argued rather than run.** The threshold
stays: Finding 12 was the last for this model. The failure is that clean fakes
and noisy real speech are not on one scale. The candidates:

1. **Put clean-audio real speech near noisy real speech on the score scale**
   by training on real speech from more recording conditions while keeping
   clean fakes in training. Findings 10 and 11 show the risk: the model learns
   each source's conditions rather than "real".
2. **Degrade the input at serving time**, e.g. a fixed codec or noise pass
   applied to every upload, so clean fakes are scored in the conditions the
   threshold was set for. Cheap to test on these same sets, but it changes
   what every reading means and must be applied identically in calibration.
3. **Report two readings** (a clean-audio and a real-world threshold, chosen
   by measured recording quality). The most honest UI, and the most machinery.

## Finding 14 — degrading every input before scoring does not help

The pre-registration below is unchanged. The result follows it, under
"Result (30 September)".

Written 30 September 2026, before anything is submitted. Option 1 from
Finding 13. No training: the Finding 9 weights are unchanged.

**Question.** If every clip passes through one fixed, deterministic channel
before the model scores it, do clean fakes stop slipping through, without
flagging more real speech?

**Candidates, fixed now** (`model.degrade_waveform`, applied after
truncation to 4 s): `none` (control), `opus` (Opus voice codec, libsndfile
compression level 0.9), `tel8k` (resample to 8 kHz and back: phone-line
bandwidth), `noise20` (white noise at 20 dB SNR, fixed seed). All four are
deterministic, checked by `verify_setup.py`.

**This is a new system, not a fifth threshold.** Each candidate changes what
the model is shown, so each needs its own threshold, fitted by the unchanged
Finding 12 procedure: People's Speech `clean` test, n 10,000, seed 42, 1%
target. The motivation comes from SpeechFake and LA (Finding 13), not from
In-the-Wild. But the weights are the ones already scored on In-the-Wild five
times, and Stage B scores them a sixth; that is disclosed, not hidden.

**Stage A — choose, without touching any test set.** For each candidate:
calibrate as above, then score SpeechFake-BD **dev**, English, at the fitted
threshold. Record fakes passed (A1), the People's Speech check-half real
flagged rate (A2), and SpeechFake dev EER (A3). The `none` control must
reproduce Finding 12's threshold (log-odds +7.98); if it does not, stop and
find out why.

- **Choice:** the candidate with the lowest A1 among those with A2 ≤ 2%.
- **Go/no-go:** Stage B runs only if the chosen A1 is at least **10 points**
  below `none`'s A1. Otherwise the finding is "no degradation helps" and
  nothing is served.

**Stage B — test the chosen one, once.** Score it on In-the-Wild, LA eval and
SpeechFake **test** (English). Serve it only if **all** hold:

- In-the-Wild EER < 5%;
- In-the-Wild real flagged ≤ 2% (today 0.62%);
- In-the-Wild fakes passed ≤ 5% (today 3.61%);
- SpeechFake test fakes passed ≤ 20% (today 32.72%).

Otherwise keep today's model and threshold, and report the numbers.

**What this cannot show.** SpeechFake dev and test hold the same 26 systems
the model trained on, so both stages are optimistic about clean fakes from
unseen systems.

### Result (30 September)

Stage A, jobs 60587095–60587102 (calibration then SpeechFake dev for each),
L40S, commit `8f2f28a`. SpeechFake dev English is 71,370 clips (6,400 real,
64,970 fake).

| Candidate | Threshold (log-odds) | A2: People's Speech real flagged, check half | A1: SpeechFake dev fakes passed | A3: SpeechFake dev EER |
| --- | --- | --- | --- | --- |
| `none` (control) | +7.98 | 0.60% | **25.49%** | 4.87% |
| `opus` | +8.15 | 0.66% | 29.54% | 4.85% |
| `tel8k` | +8.14 | 0.60% | 29.80% | 4.89% |
| `noise20` | +7.90 | 0.68% | 52.31% | 6.91% |

The control reproduced Finding 12's threshold exactly (+7.98), so the
pipeline is sound.

**Outcome: "no degradation helps".** Every candidate meets A2 ≤ 2%, so the
choice is `opus` (lowest A1 of the three), but its A1 is 4.05 points *worse*
than the control, not 10 better. Stage B was not run. In-the-Wild was not
scored. Nothing is served differently.

**Why it failed.** The hypothesis was that a shared channel would lift clean
audio up the scale towards where noisy real speech sits. It did not move real
speech: the People's Speech thresholds stayed within 0.25 log-odds of the
control. What it moved was the fakes, downwards — codec and band-limiting
blur the artefacts the model detects, and noise buries them (noise20 doubles
the miss rate and worsens ranking, EER 4.87% → 6.91%). The gap between clean
fakes and noisy real speech is in the model, not in the recording channel,
and it cannot be closed by degrading the input.

**What is left**, from Finding 13's list: retrain so that clean fakes and noisy
real speech are on one scale (option 2), or give up on a single threshold and
report per-condition readings (option 3). Both are larger pieces of work. The
served system stays as Finding 12 left it, with Finding 13's caveat on
`/result`.

## Finding 15 — a clean-audio threshold catches more fakes but accuses more real speakers

The pre-registration below is unchanged; Stage A, the amendment and Stage B
follow it.

Written 30 September 2026, before anything is submitted. Option 3 from
Finding 13. No training: the Finding 9 weights are unchanged.

**Question.** The model ranks clean fakes well (SpeechFake EER 6.26%) but its
one threshold is set for noisy real speech. If clean recordings are held to a
second threshold fitted on clean real speech, do clean-fake misses fall
without accusing more real speakers?

**The mechanism, fixed now.** `model.cleanliness_db`: the spread in dB
between a clip's loud and quiet 20 ms frames (95th minus 10th percentile), on
the same 4 s the model reads. Clips at or above a cutoff take the clean route.
`calibrate_clean.py` fits, from calibration data only and using fit halves
only:

1. the **cutoff**: the value best separating LibriSpeech test-clean (clean)
   from People's Speech (not), by balanced accuracy;
2. the **clean threshold**: 1% of LibriSpeech clips routed clean flagged, the
   same rule as every threshold since Finding 12.

The noisy route keeps today's threshold (+7.98). LibriSpeech test-clean is
held out of training: its 40 speakers share none with the 247 LibriTTS
speakers SpeechFake uses (checked by `librispeech.py check`, job 60594139).
Not held out of XLS-R's pretraining, which includes LibriVox audio.

**Stage A — choose, without touching any test set.** Fit the route on the
served checkpoint, then score SpeechFake **dev** (English) with it. Go on
only if all hold:

- SpeechFake dev fakes passed ≤ **15.49%** (the control's 25.49% minus 10);
- People's Speech check half, real flagged under routing ≤ 2%;
- LibriSpeech check half, real flagged under routing ≤ 2%.

**Stage B — once.** Score the routed checkpoint on In-the-Wild, LA eval and
SpeechFake **test** (English). Serve it only if all hold:

- In-the-Wild real flagged ≤ 2% (today 0.62%);
- In-the-Wild fakes passed ≤ 5% (today 3.61%);
- SpeechFake test fakes passed ≤ 20% (today 32.72%).

LA eval is reported, not a criterion: its real speech is VCTK, in training.

**Known weakness, stated now.** Routing can be gamed: noise added to a clean
fake sends it to the noisy route. If this is served, `/result` must say so.
The same weights are scored on In-the-Wild a sixth time in Stage B.

### Stage A result (30 September)

Jobs 60594253 (`calibrate_clean.py`) and 60594254 (SpeechFake dev, English),
L40S, commit `fca641f`.

| | Fitted / measured |
| --- | --- |
| Cutoff | 30.6 dB (balanced accuracy 87.5% on the fit halves) |
| Routed clean, check halves | LibriSpeech 67.6%, People's Speech 10.3% |
| Clean threshold | log-odds **+4.55** (noisy route stays +7.98) |
| LibriSpeech check half, real flagged | 1.23% routed (0.00% single threshold) ✓ ≤ 2% |
| People's Speech check half, real flagged | 0.71% routed (0.60% single) ✓ ≤ 2% |
| **SpeechFake dev fakes passed** | **15.94%** routed (25.49% single) ✗ needed ≤ 15.49% |
| SpeechFake dev routed clean | 72.4% of real, 79.5% of fake clips |

**Outcome: no-go, by 0.45 points.** Routing cut clean-fake misses by 9.55
points, from 25.49% to 15.94%, and both real-speech checks passed with room to
spare. But the bar was 10 points, fixed in advance, and it was missed. Stage B
has not been run.

Where the remaining misses are: BigVGAN 100% (EER 51.5%, the model cannot
separate it at all), ParlerTTS 74.9%, WaveGlow 63.2%, DiffGANTTS 59.2%,
FastSpeech2 35.8%, CosyVoice 30.3%, HifiGAN 18.0%. The other 19 of the 26
systems pass at under 12%.

### Amendment (30 September, before Stage B)

The project owner chose to run Stage B despite the missed bar. Recorded here
before any Stage B job is submitted. Reasons: the 10-point bar was a round
number chosen without a principled basis; the improvement was 9.55 points;
both real-speech checks passed; and Stage A read no test set. **Nothing else
changes:** the routed checkpoint is the one Stage A fitted, Stage B scores it
once, and its serving criteria above are applied as written. Any writeup
quoting Finding 15 must say that Stage A's go/no-go bar was missed and
overridden.

### Stage B result (30 September)

Jobs 60598372 (In-the-Wild), 60598373 (LA eval), 60598374 (SpeechFake test,
English), L40S, commit `573ff5d`, the routed checkpoint Stage A fitted.

| | Single threshold (served) | Routed | Criterion |
| --- | --- | --- | --- |
| In-the-Wild real flagged | 0.62% | **4.00%** | ≤ 2% ✗ |
| In-the-Wild fakes passed | 3.61% | 2.41% | ≤ 5% ✓ |
| SpeechFake test fakes passed | 32.72% | **23.53%** | ≤ 20% ✗ |
| LA eval fakes passed (reported) | 46.25% | 28.68% | — |
| LA eval real flagged (reported) | 0.00% | 0.01% | — |
| Routed clean: In-the-Wild real / fake | — | 60.7% / 79.4% | — |
| Routed clean: SpeechFake test real / fake | — | 71.9% / 77.8% | — |
| Routed clean: LA eval real / fake | — | 99.4% / 81.7% | — |

**Outcome: not served.** Two of three criteria fail. In-the-Wild's EER is
unchanged at 2.65% (same scores); only the operating points moved.

**Why: the cleanliness measure does not measure what matters.** It was meant to
pick out studio-clean audio, and on the calibration sets it roughly did
(LibriSpeech 67.6% routed clean, People's Speech 10.3%). But 60.7% of
In-the-Wild's genuine clips — interviews, speeches, broadcasts — also have
wide loudness range, because real speech has pauses whatever the channel.
They were held to the clean threshold (+4.55 instead of +7.98) and six times
as many were flagged. A loudness-range statistic separates *quiet pauses* from
*noisy pauses*; the model's scale separates something else, and People's
Speech was not representative of In-the-Wild in this respect (10% routed clean
against 61%).

**What it does show:** a second, stricter threshold does catch clean fakes —
SpeechFake test misses fell 9.2 points and LA's 17.6 — so the idea is sound
if the route can be chosen by the property the model actually responds to.
Picking a better routing statistic is now a search, and every candidate would
have to be judged without In-the-Wild, whose real clips were the ones it
failed on. This finding has spent the cheap options.

**Where that leaves the served system:** unchanged, Finding 12's single
threshold, with Finding 13's caveat. The remaining route to fewer clean-fake
misses is a model change (Finding 13, option 2).

## Finding 16 — training every clip through a random recording channel (pre-registered)

Written 30 September 2026, before any job is submitted. Nothing below may be
changed after In-the-Wild is scored; the result is reported whatever it is.
Option 2 from Finding 13: a model change.

**Question.** If every training clip, real and fake alike, passes through the
same random recording channel — room reverb, background noise or music, a
lossy codec — does the model stop scoring clean audio far below noisy real
speech, so that one threshold catches clean fakes without flagging more
real-world speakers?

**Why this, after Findings 10, 11 and 14.** Every training clip so far, real
and fake, has been clean studio or audiobook audio, so nothing taught the
model that noise is unrelated to the label. Findings 10–11 added noisy clips
to one class only (real), and the model learned those sources' conditions as
a sign of realness. Finding 14 degraded inputs at scoring time only, and
blurred the artefacts that weights trained on clean fakes look for. Here the
channel is drawn from one distribution for every clip whatever its label, so
it carries no information about the label, and the fakes the model learns
from are degraded as often as the real clips.

**The one change.** Finding 9's run plus `--channel-aug` (`channel_aug.py`),
applied after RawBoost to every training clip: reverb with p = 0.3 (OpenSLR 28
simulated and real room impulse responses), MUSAN noise or music with p = 0.5
at 5–25 dB SNR, and with p = 0.5 one of MP3, Opus (random quality) or 8 kHz
phone band. About 17.5% of clips pass untouched. MUSAN's speech subset is not
used. Everything else is held fixed: SSL-AASIST, RawBoost 5,
`--extra-train speechfake`, 4 epochs, batch 14, constant lr 1e-6, class
weights from the data, selection by LA dev + SpeechFake dev (both left clean),
`best.pth` by dev EER. Neither MUSAN nor the RIRs appear in any test or
calibration set. Checkpoints: `.../plus-speechfake/channel/nodp/`.

**Calibration, fixed now:** Finding 12's procedure unchanged — `calibrate.py
--calibration-set peoples_speech --target-frr 0.01`, n 10,000, seed 42.

**Stage A — without touching any test set.** Score SpeechFake **dev**
(English) at the calibrated threshold. The control is the served model at its
threshold, measured in Finding 14: 25.49% fakes passed, EER 4.87%, People's
Speech check half 0.60%.

- A1, SpeechFake dev fakes passed: must be ≤ **15.49%** (the control minus 10
  points, the same bar as Finding 15).
- A2, People's Speech check half real flagged: must be ≤ 2%.
- A3, SpeechFake dev EER: reported.

Stage B runs only if A1 and A2 both hold.

**Stage B — once.** Score the calibrated checkpoint on In-the-Wild, LA eval
and SpeechFake **test** (English). Serve it in place of Finding 12's model
only if **all** hold (the Finding 14/15 criteria):

- In-the-Wild EER < 5% (today 2.65%);
- In-the-Wild real flagged ≤ 2% (today 0.62%);
- In-the-Wild fakes passed ≤ 5% (today 3.61%);
- SpeechFake test fakes passed ≤ 20% (today 32.72%).

LA eval is reported, not a criterion (its real speech is VCTK, in training).

**Outcomes, named now:**

- **Served:** Stage B's four criteria met.
- **Improved, not served:** Stage A passed, a Stage B criterion failed.
- **No effect:** A1 > 15.49%; Stage B is not run and In-the-Wild is not read.
- **Broken:** A2 > 2%, or dev EER at selection worse than 6% (Finding 9: 3.01%).

**Expected** (written down so it can be wrong): A1 around 15%, SpeechFake dev
EER a little worse than 4.87%, In-the-Wild EER between 2% and 4%.

**Limits, stated now.** SpeechFake dev and test hold the 26 systems trained
on, so both are optimistic about unseen clean fakes. The noise, codecs and
rooms are chosen from what the literature uses, not tuned; if this fails, the
next variant has to be argued for, not tried as a quick change of p or SNR.
In-the-Wild has never scored these weights.

### Amendment (30 September, before training)

Recorded after the noise was downloaded and before any training job ran.
Nothing has been scored. The owner requires data that allows commercial use,
so the augmentation's material is narrowed; nothing else in the
pre-registration changes.

- **Noise and music:** MUSAN licenses each file separately. Only files whose
  own licence is public domain, CC BY or CC BY-SA are used (`channel_aug.py`,
  `licence_classes`): 878 of 930 noise files and 609 of 660 music tracks.
  Excluded: 32 CC BY-ND tracks, 12 whose licence the parser cannot read and
  8 with no licence entry. Checked with `channel_aug.py check` on M3.
- **Reverb:** the 60,000 simulated room impulse responses only (OpenSLR 28,
  Apache 2.0). The package's real RIRs come from other databases under their
  own terms and are not used.
- **Resources:** 16 CPUs and 64 GB for the training job rather than 8 and
  32 GB. The augmentation measured 89 ms per clip on M3, which 8 DataLoader
  workers would only just keep ahead of the GPU. This changes speed, not the
  model.

### Stage A result (1 October)

Jobs 60602024 (training, 11h43m on an H100 at 5.25 steps/s, commit `4b0ab98`),
60602025 (calibration) and 60602026 (SpeechFake dev, English), L40S. Dev EER
by epoch: 10.51%, 5.71%, 4.31%, **3.20%** (`best.pth` = epoch 4; Finding 9:
3.01%).

| | Control (Finding 12's model) | **Channel augmentation** | Bar |
| --- | --- | --- | --- |
| Threshold (log-odds, People's Speech 1%) | +7.98 | +6.60 | — |
| A2: People's Speech check half, real flagged | 0.60% | 0.56% | ≤ 2% ✓ |
| **A1: SpeechFake dev fakes passed** | 25.49% | **23.48%** | ≤ 15.49% ✗ |
| A3: SpeechFake dev EER | 4.87% | 4.61% | reported |

**Outcome: No effect.** Clean-fake misses fell by 2.01 points, not the 10
required. Stage B was not run and In-the-Wild was not scored.

What moved: the threshold fell 1.4 log-odds, so noisy real speech now scores
lower relative to everything else — the augmentation did shrink the gap it was
aimed at, a little. Ranking improved slightly (EER 4.87% → 4.61%). But the
clean fakes that pass are still the same kind: vocoder-only systems pass most
(BigVGAN 100%, HifiGAN 74%, FastDiff 63%, DiffGANTTS 56%), while full TTS
systems are mostly caught (ChatTTS 1%, FireRedTTS 0.7%, GlowTTS 0.1%). The
misses are concentrated in resynthesis by neural vocoders, which a recording
channel neither creates nor hides.

**Consequence for Finding 18**, by its pre-registered rule: Finding 16's
outcome is "No effect", so Finding 18 trains on **Finding 9's recipe** (no
channel augmentation). Nothing is served differently.

## Finding 17 — withdrawn before anything ran (MLAAD)

Pre-registered on 30 September (commit `0f04aa5`) and withdrawn the same
evening, before any data was downloaded or any job submitted. The plan added
MLAAD's English fakes (143 TTS systems) to training and held 40 of them out
by model family as an unseen-generator test. Two constraints set by the
owner exclude it: **data must allow commercial use**, and **no corpus may
require an account or sign-up**. MLAAD is CC BY-NC 4.0 and gated on Hugging
Face. The code was reverted; the pre-registration is in git history. Nothing
was measured.

## Finding 18 — our own fakes: 8 more generators to train on, 8 never heard (pre-registered)

Written 1 October 2026, after 3-clip smoke tests of each generator and before
any bulk generation, training or scoring. Nothing below may be changed after
Stage B is scored; the result is reported whatever it is.

**Question.** Finding 9's gain came from generator diversity. Does training on
fakes from 8 more open TTS families cut the share of fakes from **families the
model has never heard** that pass the threshold, without costing In-the-Wild?
And, before any training: how many such fakes does the served model miss?

**Why our own.** The project requires data that allows commercial use and
needs no account (Finding 17). No public corpus of modern fakes meets both, so
the fakes are generated here (`synth/`, `ai_model/synth.py`) with models whose
code and weights are MIT, Apache 2.0 or CC BY 4.0, all downloadable without an
account. Voices and text come from LibriSpeech (CC BY 4.0).

**The split, fixed in `synth.py` before generation** (by family, RandomState
42; Parler stays in train because SpeechFake trains on it):

| Split | Families (models) | Voices and text |
| --- | --- | --- |
| train | Kokoro, Kyutai TTS, Maya1, OuteTTS, Parler, Soprano, VibeVoice-Realtime, Zonos — 6,000 clips per family | train-clean-100 transcripts; cloning models clone its 251 speakers, who are already in training as real speech (LibriTTS via SpeechFake) |
| heldout-a (Stage A) | Chatterbox (+Turbo), Qwen3-TTS (0.6B, 1.7B), SpeechT5, VoxCPM (0.5B, 1.5) — 1,000 per family | test-clean speakers 1–20; real side: their genuine test-clean clips |
| heldout-b (Stage B) | Dia, Kitten, Marvis, Piper (3 voices) — 1,000 per family | test-clean speakers 21–40; real side likewise |

Clips per family are split evenly across its models (Piper: 334 per voice).
Changes from the first registry, made after the smoke tests and before any
bulk generation: VibeVoice uses the 0.5B streaming model, because Microsoft
withdrew the 1.5B's code; Marvis uses its transformers checkpoint, same
weights; voice prompts are 5.5–10 s (Chatterbox-Turbo refuses shorter); and
Marvis's 10-second output cap is raised with the text's length. All 17 models
passed a 3-clip smoke test with Whisper WER 0–7%, except four sibling models
(Qwen3-TTS 1.7B, VoxCPM 1.5, Piper LJSpeech and Cori), first checked at bulk
QC.

**Quality gate, fixed now.** Each model's output is checked by `synth/qc.py`:
Whisper small transcribes 50 clips and the word error rate against the
requested text is measured. A model whose **median WER exceeds 30%** is
dropped as broken, and so is any model that cannot be made to generate. A
dropped model's clips are not replaced by another model; a family with no
surviving model is removed from its split, not refilled. Every drop is
reported. Clips are never filtered one by one.

**Stage 0 — before any training (diagnostic).** Score the served model at its
threshold on `heldout-a` (the control for Stage A). `heldout-b` is not read.

**The training run.** `--extra-train speechfake+synth` (the train split's
~48,000 fakes) on top of a base recipe chosen by rule: **Finding 16's recipe
if its outcome is "Served" or "Improved, not served", otherwise Finding 9's.**
Everything else held: SSL-AASIST, RawBoost 5, 4 epochs, batch 14, constant lr
1e-6, class weights from the data, selection by LA dev + SpeechFake dev.
Calibration: Finding 12's procedure (People's Speech, 1%, n 10,000, seed 42).

**Stage A — no test set touched.** Score `heldout-a` at the calibrated
threshold. The control is Stage 0's number for the served model, or the base
model's if Finding 16's recipe is used.

- A1, heldout-a fakes passed: ≤ max(control − 10 points, control / 2).
- A2, People's Speech check half real flagged: ≤ 2%.
- Reported: heldout-a EER and real flagged, SpeechFake dev fakes passed.

**Stage B — once.** Score In-the-Wild, SpeechFake test (English), LA eval and
`heldout-b`, and the control on `heldout-b`. Serve only if **all** hold:

- In-the-Wild EER < 5%, real flagged ≤ 2%, fakes passed ≤ 5%;
- heldout-b fakes passed ≤ max(control − 10 points, control / 2);
- SpeechFake test fakes passed no worse than the served model's 32.72%.

**Outcomes:** **Served**; **Improved, not served** (Stage A passed, Stage B
failed); **No effect** (A1 misses its bar; Stage B not run); **Broken** (A2 >
2%, or dev EER at selection above 6%).

**Expected** (so it can be wrong): the served model misses 20–50% of
heldout-a fakes; training on our fakes halves that; In-the-Wild EER
2–3.5%.

**Limits, stated now.** The held-out families are open models, not the
commercial services (ElevenLabs and the like) that cannot be used here. All
fakes read audiobook text in clean conditions, so they test clean-fake
detection, the open problem, not noisy real-world fakes. The split is by
family name; related architectures may still share components (Marvis is
built on the Sesame CSM design; several models share audio codecs).
LibriSpeech test-clean was a calibration set in Finding 15, which was not
served. Chatterbox output carries Resemble's Perth watermark, as it does in
the wild.

### Amendment (1 October, during generation, before any training or scoring)

OuteTTS generates at 0.02 clips/s through its Hugging Face backend (about 55
GPU-hours for 6,000 clips), while M3 allows 4 GPUs per user. It is capped at
**2,000 clips**, the first 2,000 of the same seeded job list. The train split
is then ~44,000 fakes rather than ~48,000. Nothing else changes.

### Stage 0 result (2 October): the served model against open TTS it never heard

Job 60644034 (H100, commit after `0b89ef7`), the served checkpoint
(`best_calibrated_peoples_speech_1pct.pth`, threshold +7.98) on `heldout-a`:
4,000 fakes from 7 models in 4 families, all passing the Whisper gate (median
WER 0.0), and 1,281 genuine LibriSpeech test-clean clips from the same 20
speakers.

| Model | EER | Fakes passed at the served threshold |
| --- | --- | --- |
| Chatterbox | 3.81% | 75.20% |
| Chatterbox-Turbo | 4.21% | 81.40% |
| SpeechT5 | 4.91% | 75.50% |
| VoxCPM 1.5 | 38.62% | **100.00%** |
| VoxCPM 0.5B | 40.77% | **100.00%** |
| Qwen3-TTS 0.6B | 44.77% | **100.00%** |
| Qwen3-TTS 1.7B | 45.42% | **100.00%** |
| **Pooled** | **28.57%** | **88.45%** (real flagged 0.08%) |

**The served model does not detect modern open-source TTS it has not heard.**
Two failure modes, cleanly separated:

- **Ranking failure** (Qwen3-TTS, VoxCPM — both 2025–26 LLM-based models with
  neural codecs): EER 39–45%, near chance. The model cannot tell these fakes
  from real speech at any threshold. No threshold, routing or calibration can
  fix this; only training data that teaches what they sound like can.
- **Threshold failure** (Chatterbox, SpeechT5): EER 4–5%, the ranking works,
  but three quarters still score below a threshold set for noisy real-world
  speech — Finding 13's clean-audio problem again.

This is Stage 0's control for Finding 18's Stage A: **A1 control = 88.45%**,
so the pre-registered bar is ≤ max(88.45 − 10, 88.45 / 2) = **78.45%**.

This set is now shown on `/result` and the home page (copied beside the
served checkpoint as `best.measured-synth.json`): the served model was never
trained, selected or calibrated on it, so it is an honest measured rate.

### Training and Stage A (3 October): passed

Finding 16's outcome was "No effect", so the base recipe is Finding 9's.
Training job 60661487 (H100, 12 h 40 min, `--extra-train speechfake+synth`):
dev EER fell every epoch (5.72%, 4.46%, 2.89%, **2.19%**) so `best.pth` is
epoch 4, well under the 6% "Broken" bar. Checkpoints:
`.../plus-speechfake+synth/nodp/`.

Calibration, job 60689639 (Finding 12's procedure): threshold P(spoof) =
0.999153, log-odds **+7.07**; People's Speech fit half 1.01%, **check half
1.05%** flagged. A2 (≤ 2%) passes.

Stage A, job 60694935, `heldout-a` at that threshold (same 5,281 clips as
Stage 0):

| Model | EER, control → new | Fakes passed, control → new |
| --- | --- | --- |
| Chatterbox | 3.81% → 1.80% | 75.20% → 1.80% |
| Chatterbox-Turbo | 4.21% → 2.19% | 81.40% → 2.40% |
| SpeechT5 | 4.91% → 1.40% | 75.50% → 0.90% |
| VoxCPM 1.5 | 38.62% → 24.24% | 100.00% → 67.00% |
| VoxCPM 0.5B | 40.77% → 24.42% | 100.00% → 64.00% |
| Qwen3-TTS 0.6B | 44.77% → 28.23% | 100.00% → 83.80% |
| Qwen3-TTS 1.7B | 45.42% → 24.99% | 100.00% → 78.40% |
| **Pooled** | **28.57% → 16.78%** | **88.45% → 37.40%** (real flagged 0.08% → 1.64%) |

**A1 passes**: 37.40% against a bar of 78.45%. Reported, job 60694936:
SpeechFake dev (English) fakes passed **10.94%** (control 25.49%, Finding
14), EER 3.95%.

What it says, before Stage B: the threshold-failure families (Chatterbox,
SpeechT5) are essentially solved — their ranking was already good, and
training on clean open-TTS fakes moved them above the threshold. The
ranking-failure families (Qwen3-TTS, VoxCPM) improved from chance to EER
~25%, but two thirds or more still pass: training on other modern TTS
teaches only part of what these LLM-codec models sound like. Expected was
"halves the misses"; the pooled rate fell by 58%.

Stage B is submitted next, unchanged from the registration above.

### Stage B (3 October): improved, not served — one bar missed by 0.18 points

Jobs 60695901–60695905, each set scored once at the calibrated threshold
(+7.07). Served model (threshold +7.98) beside it for reference.

| Test set | EER, served → new | Real flagged, served → new | Fakes passed, served → new |
| --- | --- | --- | --- |
| In-the-Wild | 2.65% → **2.02%** | 0.62% → **2.18%** | 3.61% → **1.89%** |
| SpeechFake test (en) | 6.26% → 4.15% | 0.01% → 0.00% | 32.72% → 13.65% |
| LA eval | 2.12% → 0.53% | 0.00% → 0.00% | 46.25% → 20.65% |
| heldout-b (control run, job 60695905) | 12.09% → 7.70% | 0.00% → 3.51% | 73.86% → 15.27% |

heldout-b by model, fakes passed control → new: Dia 65.9% → 26.1%, Kitten
91.5% → 0.0%, Marvis 42.4% → 0.2%, Piper Cori 99.7% → 49.4%, Piper
LibriTTS-R 98.5% → 9.0%, Piper LJSpeech 88.3% → 45.8%. All 17 models passed
the Whisper gate (Marvis on its second generation run, job 60661544); none was
dropped.

Serving criteria:

- In-the-Wild EER < 5%: **2.02%, pass.**
- In-the-Wild real flagged ≤ 2%: **2.18%, fail** (436 of 19,963 clips; 2%
  would be 399).
- In-the-Wild fakes passed ≤ 5%: **1.89%, pass.**
- heldout-b fakes passed ≤ max(73.86 − 10, 73.86 / 2) = 63.86%: **15.27%, pass.**
- SpeechFake test fakes passed ≤ 32.72%: **13.65%, pass.**

**Outcome: Improved, not served** (Stage A passed, Stage B failed on one
criterion). The served model is unchanged.

What it says: training on 8 more open-TTS families improved every
measurement except one. Ranking improved on all four test sets (In-the-Wild
EER 2.65% → 2.02%), clean fakes passed fell by half or more everywhere, and
fakes from families it never heard fell from 74–88% passed to 15–37%. The cost
is the real side: the same 1% People's Speech target flags 2.18% of
In-the-Wild real speech (from 0.62%) and 3.51% of LibriSpeech test-clean (from
0.00%). The new model is more suspicious of clean speech, real or fake, so
calibrating on noisy People's Speech no longer carries over to cleaner real
audio as well — Finding 13's scale problem, now on the real side.

Expected was "halves the misses, In-the-Wild EER 2–3.5%"; both held.

### Override (3 October, written before the checkpoint was swapped)

**The owner overrides the In-the-Wild real-flagged bar and serves the Finding
18 model.** The bar was missed by 0.18 points (2.18% against 2%); every other
criterion passed with a wide margin. The trade, stated plainly: about 1.6 more
genuine In-the-Wild clips flagged per 100 (0.62% → 2.18%), and LibriSpeech
test-clean real flagged 0.00% → 3.51%, in exchange for fakes passed falling
on every test set — In-the-Wild 3.61% → 1.89%, SpeechFake test 32.72% →
13.65%, LA eval 46.25% → 20.65%, unseen open TTS 74–88% → 15–37%. The
pre-registered outcome stays "Improved, not served"; the served model is
served by override, and any writeup must say so, as with Finding 15.

Served: `.../plus-speechfake+synth/nodp/best_calibrated_peoples_speech_1pct.pth`
(threshold +7.07). The `/result` unseen-TTS row is `heldout-b` (job
60695904): it played no part in training, selection, calibration or the
Stage A gate, unlike `heldout-a`. The previous served checkpoint is kept on M3
unchanged.

## What has not been measured

- **AASIST under DP.** The port trains under Opacus, but the private AASIST
  run has not been done, so the cost of privacy exists only for the CNN.
- **AASIST with BatchNorm, or over several seeds** — the two cheapest ways to
  find out how much of the 3.17%-vs-0.83% gap comes from the GroupNorm swap.
- **The rest of the comparison table** in `APPROACH.md`: LCNN-LSTM-sum and
  AASIST-L. (The SSL front-end is Finding 8.)
- **SSL-AASIST under DP.** Full DP fine-tuning of 316M parameters is
  expensive; the practical version freezes XLS-R and trains only the back-end
  privately. Not designed yet.
- **A per-epoch sweep of the SSL runs**, to see how arbitrary `best.pth` is
  once LA dev has saturated.
- **More SpeechFake epochs**, or SpeechFake without RawBoost — dev EER was
  still falling at epoch 4, and the two changes were not separated.
- **A DP sweep.** One ε is a point, not the cost-of-privacy curve.
- **Any tuned run.** Five epochs, one learning rate, one batch size, throughout.
- **RawBoost and ASVspoof 5 together**, or RawBoost with algo 3 (upstream's
  choice for codec-compressed audio), or ASVspoof 5 at more epochs.
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
sbatch --time=14:00:00 hpc/train.slurm --arch aasist --no-dp --rawboost 5         # Finding 7
sbatch hpc/get_asvspoof5.slurm                                                    # ~3h, 58 GB
sbatch --time=20:00:00 hpc/train.slurm --arch aasist --no-dp --extra-train asvspoof5 --epochs 12
sbatch --gres=gpu:L40S:1 hpc/evaluate.slurm --dataset itw --arch aasist --rawboost 5
sbatch --partition=m3h --qos=m3h --gres=gpu:H100:1 --time=24:00:00 \
       hpc/train.slurm --arch ssl-aasist --no-dp --rawboost 5                    # Finding 8
sbatch --gres=gpu:L40S:1 hpc/evaluate.slurm --dataset itw --arch ssl-aasist --rawboost 5
sbatch hpc/get_speechfake.slurm                                                   # ~3.5h, 290 GB
sbatch --partition=m3h --qos=m3h --gres=gpu:H100:1 --time=24:00:00 \
       --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
       hpc/train.slurm --arch ssl-aasist --no-dp --rawboost 5 --extra-train speechfake --epochs 4
sbatch --partition=m3h --qos=m3h --gres=gpu:H100:1 --time=24:00:00 \
       --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
       hpc/train.slurm --arch ssl-aasist --no-dp --rawboost 5 --extra-train speechfake \
       --extra-bonafide commonvoice --epochs 4                                   # Finding 10
sbatch --gres=gpu:L40S:1 --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
       hpc/calibrate.slurm --ckpt <run-dir>/best.pth --commonvoice-split test \
       --out <run-dir>/best_calibrated_commonvoice_test.pth
sbatch --gres=gpu:L40S:1 --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
       hpc/evaluate.slurm --dataset itw --ckpt <run-dir>/best_calibrated_commonvoice_test.pth
# Pin evaluation to a 48GB+ GPU: at batch 128, AASIST runs out of memory on a T4.
cd ai_model && python summarise_results.py <run-dir>
```
