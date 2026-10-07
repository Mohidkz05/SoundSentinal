# SoundSentinal — FYP B write-up and presentation plan

Started 7 October 2026; restructured the same day against the two marking
rubrics (`FYP_B_Final_Paper_Rubric.pdf`, `FYP B Final Presentation
Rubric.pdf`). An outline, not the paper: each section says what it argues,
which evidence carries it, and where that evidence lives. Every number is
copied from `RESULTS.md`, which is the record; re-check against it before
submission.

## What the rubrics require

**Paper** — weighted 30 / 50 / 10 / 10:

| Category | Weight | What earns HD |
| --- | --- | --- |
| Background, literature, research question | 30% | Current *and* seminal literature, grey literature where relevant, critical evaluation that finds gaps, a research question justified by the review |
| Scientific and engineering content | 50% | Clear method with figures/tables; results with statistical analysis where applicable, no overclaiming, failures reported so others can learn; discussion linked back to the literature, bias and error addressed, limitations and future work; originality |
| Structure, figures, tables | 10% | **The provided template, strictly. Over 10 pages is an N.** Abstract, Introduction, Conclusion mandatory; figures cross-referenced |
| Clarity and expression | 10% | Terms defined, one referencing style, **AI use acknowledged** |

**Presentation** — 10 minutes, hard stop, for a *general engineering
audience*. It is **not** a summary of the paper. Weighted: technical content,
constraints and project plan 60%; structure and visuals 15%; answers to
questions 15%; individual delivery 10%. It must cover why anyone should
care, who benefits, project management and setbacks, why these methods, and
constraints such as safety, whole-life cost, net zero carbon and
socio-environmental impact.

## Decisions (7 October 2026)

1. **Template: LaTeX, IEEE Transactions (`IEEEtran` journal).** The paper is
   `paper/main.tex` with references in `paper/refs.bib` (IEEE style via
   `IEEEtran.bst`). Build with `~/tools/tectonic/tectonic paper/main.tex` or
   on Overleaf. Red `[TODO: …]` markers show what is left; there must be none
   at submission. Slides use Monash's `powerpoint-template-standard.pptx`.
2. **Framing: generalisation first, DP second.** Research question:
   > *Can a deepfake speech detector trained on public data be made to work on
   > real-world recordings, and how should its output be reported so that a
   > non-expert is not misled?*
   with *what does DP-SGD cost such a detector?* as the secondary question.
3. **Team submission**: Muhammad Mohid Khanzada, Amaan Muhammad, Farhan
   Mohammed, in that author order. Agreed split (7 October):

   | | Paper | Slides (presents his own) |
   | --- | --- | --- |
   | Mohid | I (drafted), III Method, IV Results, V–VII; final edit to 10 pages | Demo, how it works, why these choices, the journey (37% → 2%) |
   | Amaan | II-A–II-C (benchmarks, front-ends, back-ends); verify all references | The problem and why it matters; constraints (cost, carbon, privacy) |
   | Farhan | II-D–II-F (generalisation, DP, deployed detectors); Figs 2 and 3 | Project management and timeline; prepared Q&A answers |

   Briefs: `paper/briefs/amaan.md`, `paper/briefs/farhan.md`. Git also shows
   commits from Alex Ung (August 2025); confirm whether to acknowledge him.
4. **AI acknowledgement**: a section in the paper, wording still to agree.

---

# Part 1 — the paper (10 pages)

Page budget in brackets, figures included. Rubric weight in bold where a
section carries it.

## Abstract (in page 1)

~200 words: the problem, the system, In-the-Wild EER **2.02%**, real
recordings flagged **2.18%** and fakes missed **1.89%** at the served
threshold, the clean-fake weakness (13.7–20.7% missed), the DP cost (+7.65
points of EER at ε = 0.48 on the CNN).

## 1. Introduction (≈0.75 page)

- Why it matters: voice-cloning fraud; the person with one clip and one
  question.
- The gap, in one paragraph: benchmark EERs near 1% collapse on real-world
  audio (Müller et al.), and a single accuracy figure hides two errors with
  very different costs (accusing a real speaker vs. missing a fake).
- Research question (decision 2) and contributions:
  1. a detector at 2.02% EER on In-the-Wild, never trained, selected or
     calibrated on it;
  2. evidence that the decision threshold, not the ranking, was the hard
     problem, and a calibration procedure on held-out real speech;
  3. a pre-registered protocol: outcomes and bars written before each run;
  4. a measured cost of DP-SGD on a small detector;
  5. an interface that reports a reading against a visible threshold.

## 2. Background and related work (≈2 pages) — **30% of the mark**

The heaviest-weighted category gets the most room relative to its length.
Each subsection ends with the gap it leaves, which is what "critical
analysis" means here.

- **2.1 Benchmarks and metrics.** ASVspoof 2015 → 2019 LA → 2021 → 5;
  EER vs. min t-DCF and why t-DCF is primary. *Gap:* both are pooled,
  threshold-free numbers; neither says what happens at a deployed threshold.
- **2.2 Front-ends.** CQCC and LFCC baselines; log-Mel and why the Mel scale
  compresses where vocoder artefacts live; raw waveform (SincNet / RawNet2);
  self-supervised (wav2vec 2.0, XLS-R). Seminal + current.
- **2.3 Back-ends.** LCNN, RawGAT-ST, AASIST, SSL-AASIST. *Inconsistency:*
  RawNet2's reported LA EER ranges 0.99–9.5% across sources (`APPROACH.md`).
- **2.4 Generalisation.** In-the-Wild's collapse result; RawBoost; newer
  multi-generator corpora (SpeechFake, MLAAD — and why MLAAD was excluded on
  licence grounds). *Gap:* most papers report EER only, so whether a threshold
  transfers is untested.
- **2.5 Differential privacy.** DP-SGD (Abadi et al.), Opacus, the PRV
  accountant. *Gap:* no published cost-of-privacy figure for spoofing
  countermeasures; and a critical point — DP protects the training speakers of
  a *public* corpus, not the user's uploaded clip.
- **2.6 Deployed detectors (grey literature).** Commercial and free detectors
  report a verdict or one accuracy figure, never a threshold or per-error
  rates (`DESIGN.md`, competitor table; company pages as grey literature).
- Close with the research question, justified by 2.1, 2.4 and 2.6.

## 3. Method (≈2 pages) — part of the 50%

- **3.1 Data** **[Table 1]**: every corpus, its role (train / select /
  calibrate / evaluate only) and licence. In-the-Wild evaluation-only;
  heldout-a and heldout-b own-generated families; the VCTK overlap that
  makes LA eval no longer cleanly held out once SpeechFake is in (Finding 9).
- **3.2 Models** **[Fig 1: pipeline diagram]**: 16 kHz mono, first 4 s →
  CNN (267k, log-Mel/LFCC) / AASIST (297k, GroupNorm, deviations listed in
  `aasist.py`) / SSL-AASIST (XLS-R 300M, ~316M params). One `model.py` for
  training and serving, with the parity test (`verify_setup.py`).
- **3.3 Training**: recipes per architecture; class weights; RawBoost; DP
  settings (noise 1.1, clip 1.0, δ = 1e-5, ε = 0.48). Say the CNN and AASIST
  rows differ by schedule as well as architecture.
- **3.4 Calibration**: scores in log-odds (softmax saturates); threshold so 1%
  of held-out People's Speech real clips is flagged; uncertain band at the 95th
  percentile.
- **3.5 Evaluation protocol**: eval partitions only, never dev; EER, min
  t-DCF, per-attack EER, and both error rates at the threshold, with Wilson
  95% intervals. **Pre-registration**: outcome categories and serving bars
  written in `RESULTS.md` before each job; In-the-Wild scored once per model;
  overrides written down before the next run.

## 4. Results (≈2.5 pages) — part of the 50%

Every result with its interval where one exists; failures reported in the
same voice as successes.

- **4.1 Benchmark** **[Table 2]**: the `RESULTS.md` Summary table, trimmed to
  the rows that make the argument. SSL-AASIST + RawBoost 0.79% EER / 0.0143
  min t-DCF; our AASIST 3.17% vs published 0.83% (GroupNorm, single seed,
  Finding 5); dev EER selects the wrong model (Finding 1); LFCC and log-Mel
  are complementary per attack (Finding 2).
- **4.2 The cost of privacy** **[Fig 2: per-attack EER, private vs not]**:
  10.15% → 17.80% eval EER (+7.65); A07–A16 almost untouched, A17–A19 at
  chance. Caveats that inflate the gap: untuned, strict ε, 5 epochs, one seed.
- **4.3 Real-world collapse and recovery** **[Fig 3: In-the-Wild EER by
  model — the main figure]**: AASIST 37.15% (Finding 6); RawBoost and ASVspoof 5
  don't fix it (Finding 7); SSL + RawBoost 11.21% (Finding 8); + SpeechFake
  2.65% (Finding 9); + own fakes 2.02% (Finding 18).
- **4.4 The threshold** **[Table 3: Findings 10–16, idea / bar / result /
  served?]** **[Fig 4: score distributions with the threshold drawn]**: real
  speech in training broke it (10–11); a 1% target worked on real-world audio
  (12); clean fakes then passed (13: 32.7%); three fixes failed (14–16).
- **4.5 The served system** **[Table 4: four test sets, with 95% CIs]**:

  | Set | Real flagged | Fakes missed |
  | --- | --- | --- |
  | In-the-Wild | 2.18% (1.99–2.40) | 1.89% (1.66–2.15) |
  | SpeechFake test (en) | 0.00% | 13.65% (13.50–13.81) |
  | ASVspoof 2019 LA eval | 0.00% | 20.65% (20.34–20.97) |
  | heldout-b (never heard) | 3.51% (2.65–4.64) | 15.27% (14.19–16.42) |

  The override: Finding 18 missed its pre-registered 2% bar by 0.18 points and
  was served anyway; state it here, in the body. Finding 19: codec
  resynthesis already flagged at 98–100%, so Qwen3-TTS and VoxCPM evade it some
  other way (64–84% still pass).

## 5. Discussion (≈1 page) — part of the 50%

Link each point back to Section 2.

- Why a pretrained front-end generalises when from-scratch models don't
  (back to 2.2–2.4).
- One threshold, two kinds of audio: the trade-off the 1% target makes.
- **Bias and error**: single seeds; intervals cover clip sampling only, and
  In-the-Wild clips share speakers, so they are optimistic; model selection on
  a dev set that reuses training attacks; LA eval contaminated by VCTK once
  SpeechFake is added; the 2.18% miss sits at the edge of its interval
  (1.99–2.40%), so the bar was missed only just measurably.
- Pre-registration as a control on researcher bias, and what the two
  overrides (Findings 15 and 18) cost in credibility.
- DP: what one point does and doesn't show (back to 2.5).
- What another researcher can learn from the failures (rubric wording).

## 6. Limitations and future work (≈0.5 page)

LLM-codec TTS (Qwen3-TTS, VoxCPM); clean real speech flagged more (LibriSpeech
0% → 3.5%); first 4 s only; not speaker verification; one seed, no EER
intervals; DP only on the CNN, no ε sweep. Future: whole-clip scoring, one
scale for clean and noisy audio, a frozen-front-end DP SSL-AASIST, a held-out
attack set for model selection, saved per-clip scores for bootstrap EER
intervals.

## 7. Conclusion (≈0.25 page) — mandatory

The answer to the research question in three sentences, the headline numbers,
and the one weakness a user must know.

## Acknowledgements, AI use, references (≈0.5 page)

Supervisor; Monash M3 (project df37); dataset licences; the AI-use statement
(decision 4). One referencing style throughout.

## Figures and tables to make

| # | What | From |
| --- | --- | --- |
| Fig 1 | Pipeline: upload → 16 kHz/4 s → model → log-odds → threshold → reading | new diagram |
| Fig 2 | Per-attack EER, CNN private vs non-private | Finding 3 JSONs |
| Fig 3 | In-the-Wild EER by model, in order of the findings | Findings 6–9, 18 |
| Fig 4 | Score distributions, real vs fake per set, threshold marked | needs per-clip scores (rescore on M3) or drop |
| Fig 5 | `/result` screenshot | live site |
| Table 1 | Datasets, roles, licences | Section 3.1 |
| Table 2 | Benchmark comparison | `RESULTS.md` Summary |
| Table 3 | Findings 10–16 at a glance | `RESULTS.md` |
| Table 4 | Served system with 95% CIs | above |

Fig 4 is the only item that needs new compute: `evaluate.py` doesn't save
per-clip scores. It is evaluation, not training, but training was closed on
4 October, so it is the owner's call.

---

# Part 2 — the presentation (10 minutes, hard stop)

For a general engineering audience; the paper's detail stays in the paper.
Approximate timing:

| Min | Slide(s) | Content | Rubric item |
| --- | --- | --- | --- |
| 0:00 | Hook | Play the two sample clips (one real, one synthetic); ask the room which is fake | why care |
| 1:00 | The problem | Voice-clone scams; who is harmed; why "98% accurate" claims mislead | relevance, who benefits |
| 2:00 | What we built | Live demo of the site: upload → reading against the threshold → error rates | solution, visuals |
| 3:30 | How it works | One diagram: audio → pretrained speech model → score → threshold. No jargon | method for a general audience |
| 4:30 | Why these choices | Why SSL-AASIST over the CNN; why a threshold and error rates, not a verdict | justification |
| 5:30 | The journey | Benchmark 3% → real world 37% → 2%: the collapse and the fix, one chart | setbacks and how they were addressed |
| 6:30 | Project management | Timeline Aug–Oct; M3 GPU access as the critical path; pre-registration as a project control; 19 findings, what was dropped and why | project plan, adaptability |
| 7:30 | Constraints | Cost (A$0.53 of student credit to host; scale-to-zero); carbon (H100 hours for training vs CPU inference); privacy (clips never stored); data licences; harm from false accusations | contextual factors |
| 8:30 | Honest limits | Clean studio fakes and two new TTS systems get past it; the override, said plainly | no overclaiming |
| 9:15 | Close | Who benefits, what's next | impact |

**Prepared answers** (15% of the mark): why not just use a commercial
detector; why the threshold is 1% and not something else; what the override
means for trusting the results; why DP was dropped for the served model; how
much training cost (GPU hours, carbon); could the detector be used to make
better fakes; what happens with a clip longer than 4 seconds.

**Compute used** (from `sacct` on M3, 1 August – 7 October 2026): **228.7
GPU-hours** in 175 jobs: 87.7 h on H100s (`m3h`, 12 jobs) and 141.0 h on the
general `gpu` partition (L40S / A100 / A40 / T4, 163 jobs). A rough energy
estimate, to be stated as one: at board power (H100 ~700 W, the rest taken
at ~350 W) that is about 110 kWh of GPU energy, ~150 kWh with a data-centre
overhead of 1.4; at Victoria's grid intensity (~0.8 kg CO₂e/kWh) roughly
**90–125 kg CO₂e** for all training and evaluation. Serving costs almost
nothing by comparison: CPU-only, scale-to-zero, A$0.53 of credit to date.
Check the grid factor against the current Australian National Greenhouse
Accounts Factors before quoting it.
