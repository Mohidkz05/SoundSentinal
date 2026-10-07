# SoundSentinal — write-up outline

Started 7 October 2026. An outline, not the write-up: each section says what
it argues, which evidence carries it, and where that evidence lives. Every
number below is copied from `RESULTS.md`; re-check against it before anything
is submitted, since that file is the record. Word counts are placeholders until
the required format is known.

**Open decisions (owner's call, before drafting):**

1. **The framing.** `APPROACH.md` set the research question as *what does
   differential privacy cost an audio deepfake detector?* The work since
   mid-September was mostly about something else: whether a detector trained on
   benchmark data holds up on real-world audio. The DP cost was measured once,
   on the CNN. Two honest options:
   - **A. Generalisation first, DP as a secondary result** (recommended: Findings 6–19,
     all but one of the last fourteen, are about this). Title along the lines of *"Calibrated
     deepfake audio detection: from benchmark to real-world recordings"*.
   - **B. Keep DP as the headline** and present the generalisation work as what
     building a usable detector required. Weaker, because the DP result is one
     point on one small model.
2. **The required format**: length, section list, template, referencing style.
3. **Who wrote what**, if this is a group submission (a collaborator was given
   write access on 29 September).

---

## 0. Abstract (~250 words)

- Problem: synthetic speech is cheap and convincing; most detectors are trained
  and scored on clean benchmark audio and report a single accuracy.
- What was built: SSL-AASIST (XLS-R 300M + AASIST) with RawBoost, trained on
  ASVspoof 2019 LA + SpeechFake + ~44,000 fakes we generated from 8 open TTS
  families; served through a web app that shows a score against a calibrated
  threshold and the model's measured error rates.
- Headline numbers: In-the-Wild EER **2.02%**; at the served threshold
  **2.18%** of real recordings flagged and **1.89%** of fakes missed.
- The honest caveat: clean studio-quality fakes are missed 13.7–20.7% of the
  time, and two recent TTS systems (Qwen3-TTS, VoxCPM) mostly pass.
- DP result (if framing A): one measurement, +7.65 points of eval EER at ε=0.48
  on the CNN.

## 1. Introduction (~1,000 words)

- Motivation: voice-cloning scams, the everyday user with one clip and one
  question (`PRODUCT.md`, "Users").
- Gap: benchmark EERs (~1% on ASVspoof 2019 LA) do not transfer to real-world
  audio; a single "accuracy" figure hides the two kinds of error, which cost
  different things (a false accusation vs. a missed fake).
- Contributions, as a numbered list:
  1. A detector reaching 2.02% EER on In-the-Wild, never trained, selected or
     calibrated on it.
  2. A threshold calibration procedure on held-out real speech, and evidence
     that the threshold, not the ranking, was the hard problem (Findings 9–16).
  3. A pre-registered experimental protocol: outcomes and serving bars written
     down before each job ran.
  4. A measured cost of differential privacy on a small detector (Finding 3).
  5. An interface that reports a reading against a visible threshold, with
     the live model's error rates beside it.
- Structure of the report.

## 2. Background and related work (~1,500 words)

- **Spoofing countermeasures**: the ASVspoof challenges (2019 LA, 2021, 5);
  EER and min t-DCF, and why t-DCF is the primary metric (`RESULTS.md`, "How
  to read these numbers"; `APPROACH.md`, "The comparison table").
- **Front-ends**: hand-crafted (CQCC, LFCC), log-Mel, raw waveform (SincNet),
  self-supervised (wav2vec 2.0 / XLS-R). Why the Mel scale discards
  high-frequency vocoder evidence.
- **Back-ends**: LCNN, RawNet2, RawGAT-ST, AASIST (Jung et al., ICASSP 2022);
  SSL-AASIST (Tak et al., 2022).
- **Generalisation**: In-the-Wild (Müller et al., 2022) and its finding that
  benchmark-trained models collapse; RawBoost (Tak et al., 2022).
- **Differential privacy**: DP-SGD (Abadi et al., 2016), Opacus, the PRV
  accountant; what DP protects here (training speakers) and what it does not
  (the uploaded clip) — `CLAUDE.md`, "Question DP-SGD's premise".
- **Calibration and thresholds**: why a probability from a saturated model is
  not a usable output; log-odds scores.
- **Existing products** (one paragraph, no marketing): verdict-style outputs,
  no published thresholds or per-error rates (`DESIGN.md`, competitor table).

## 3. Data (~800 words)

- Table of every dataset, its role and licence:
  - ASVspoof 2019 LA: train / dev / eval (ODC-By).
  - SpeechFake: training + dev selection (CC BY 4.0); note it trains on VCTK,
    LA's bona fide source, so LA eval is no longer cleanly held out (Finding 9).
  - Own generated fakes, 8 open TTS families for training; **heldout-a**
    (Chatterbox, SpeechT5, Qwen3-TTS, VoxCPM) gated Stage A; **heldout-b** (Dia,
    Kitten, Marvis, Piper) never used for anything but the final score
    (Finding 18).
  - People's Speech: calibration only; LibriSpeech test-clean: calibration
    diagnostics only, speaker-disjoint.
  - Common Voice, VoxPopuli: tried as bona fide training data (Findings 10–11).
  - In-the-Wild: **evaluation only**, never trained, selected or calibrated on.
- The licence rule: commercial use allowed, no NC/ND, no gated datasets; MLAAD
  rejected on those grounds (Finding 17).
- Class imbalance (~1:9 bona fide:spoof on LA) and inverse-frequency loss
  weights `[4.92, 0.56]`; why not a weighted sampler under Opacus.

## 4. Method (~2,000 words)

- **Pipeline**: mono, 16 kHz, first 4 s (64,000 samples). One `model.py`
  shared by trainer, evaluator and server, and the train/serve parity check
  (`verify_setup.py`) — with the bug that motivated it (log-Mel vs raw Mel
  silently diverging).
- **Models compared**: the 267k CNN (log-Mel, LFCC), AASIST (297k, GroupNorm
  instead of BatchNorm for DP compatibility — list the deviations from
  upstream in `aasist.py`), SSL-AASIST (XLS-R 300M front-end, ~316M params).
- **Training recipes** (`TRAIN_DEFAULTS`): CNN 5 epochs / batch 64 / 1e-3;
  AASIST 100 epochs / batch 24 / 1e-4 with cosine annealing; SSL runs on an
  H100 on Monash M3. Say plainly that the AASIST and CNN rows differ by
  schedule as well as architecture.
- **Augmentation**: RawBoost (training only); random recording channel
  (Finding 16).
- **DP**: Opacus, noise multiplier 1.1, max grad norm 1.0, δ=1e-5, ε=0.48.
- **Model selection** and its known flaw: `best.pth` by dev EER, which reuses
  the training attacks (Finding 1).
- **Calibration**: threshold set so a fixed share (1%) of held-out People's
  Speech real clips is flagged; scores in log-odds because the model saturates
  float32 softmax; the uncertain band at the 95th percentile of held-out real
  scores.
- **Evaluation protocol**: eval partition only, never dev; EER, min t-DCF,
  per-attack EER, and both error rates at the served threshold.
- **Pre-registration**: each experiment from Finding 10 on had its outcome
  categories and serving bars written in `RESULTS.md` before submission;
  In-the-Wild scored once per model. Owner overrides were written down before
  the next job ran.

## 5. Results (~3,000 words — the core)

Figures and tables to make are marked **[fig]** / **[table]**.

### 5.1 The benchmark comparison (ASVspoof 2019 LA eval)

- **[table]** the Summary table in `RESULTS.md`: published baselines vs ours.
- CNN log-Mel 9.60% (best epoch) ≈ CQCC-GMM 9.57%; our AASIST 3.17% vs the
  published 0.83% (Finding 5: GroupNorm, single seed, `best.pth` ranked 23rd
  of 100 epochs); SSL-AASIST + RawBoost 0.79% EER, 0.0143 min t-DCF — better
  than published AASIST.
- Finding 1: dev EER picks a worse model than an earlier epoch (log-Mel 0.24%
  dev vs 10.15% eval).
- Finding 2: front-ends are complementary — LFCC wins 9 of 13 attacks, log-Mel
  wins A18/A19. **[fig]** per-attack EER heatmap.

### 5.2 The cost of privacy (Finding 3)

- **[table]** non-private vs DP: eval EER 10.15% → 17.80% (+7.65 points);
  best epochs 9.60% → 17.57% (+7.97).
- DP did not degrade evenly: A07–A16 stay at 0.39–1.08%, A17–A19 go to chance
  (43–45%). **[fig]** per-attack bars, both regimes.
- Caveats that bias the gap upwards: untuned hyperparameters, very strict ε,
  five epochs, single seed.

### 5.3 The real-world collapse and the fix (Findings 6–9)

- Finding 6: every benchmark-trained model collapses on In-the-Wild (AASIST
  37.15%).
- Finding 7: RawBoost and ASVspoof 5 data do not fix it.
- Finding 8: a pretrained SSL front-end does, but only with RawBoost: 11.21%.
- Finding 9: adding SpeechFake: **2.65%**.
- **[fig]** In-the-Wild EER by model, one bar each — the project's main plot.

### 5.4 The threshold was the hard part (Findings 4, 9–16)

- Finding 4: a dev-set threshold does not survive the partition change.
- Findings 10–11: real-world bona fide speech in training broke or did not
  move the threshold.
- Finding 12: a 1% target on People's Speech — ITW 0.62% real flagged, 3.61%
  fakes missed.
- Finding 13: clean modern fakes pass (32.7% on SpeechFake): clean and noisy
  audio sit on different parts of the scale.
- Findings 14–16: three attempts to close that gap — degrading inputs, a
  second "clean" threshold, channel augmentation — none served. **[table]**
  idea / bar / result / served?
- **[fig]** score distributions: ITW real, LibriSpeech real, SpeechFake fakes,
  with the threshold drawn.

### 5.5 Training on our own fakes (Findings 18–19)

- Finding 18: 8 more open TTS families in training. Every number improved
  (**[table]** the four-row served table, old vs new), but ITW real flagged
  2.18% missed the pre-registered 2% bar by 0.18 points.
- **The override**: served anyway by the owner on 3 October, written down
  before the swap. State it in the body, not a footnote.
- Finding 19: codec-resynthesised real speech — no room, the model already
  flags 98–100% of it; so Qwen3-TTS / VoxCPM evade it by something other than
  a codec fingerprint.

## 6. The application (~1,200 words)

- Architecture: Next.js on Cloudflare Workers → proxy route → Flask/PyTorch on
  Azure Container Apps (scale to zero, bearer token). Video decoded in the
  browser; clips processed in memory, never stored.
- Design thesis: *an instrument, not a verdict machine*. The score on a
  log-odds scale, the threshold drawn, the uncertain band, confidence as
  certainty not accuracy, the live error rates beside every reading. Why the
  axis is log-odds (P = 0.99966 at the threshold). **[fig]** screenshot of
  `/result`.
- Accessibility and performance: WCAG 2.2 AA; the verdict palette checked
  under colour-blindness simulation (teal↔ember fails, ΔE 0.048); Lighthouse
  mobile 90–97, CLS 0; works without WebGL.
- Measured serving: ~1.2 s per prediction warm, 28.8 s cold.

## 7. Discussion (~1,200 words)

- Why the SSL front-end generalises and the from-scratch models do not.
- One threshold for two kinds of audio: the clean-vs-noisy scale problem, and
  the trade-off the 1% target makes (fewer false accusations on real-world
  audio, more missed clean fakes).
- What the DP measurement does and does not show; the case that DP protects
  the wrong party for a public training corpus.
- Pre-registration in a student project: what it prevented (tuning to
  In-the-Wild) and what the two overrides cost in credibility.

## 8. Limitations and future work (~800 words)

From `RESULTS.md` "What has not been measured" and the served-system section:

- Recent LLM-codec TTS mostly passes (Qwen3-TTS 78–84%, VoxCPM 64–67%).
- Clean real speech is flagged more (LibriSpeech 0% → 3.5%).
- Only the first 4 seconds are read; not speaker verification.
- Every result is a single seed; no error bars.
- DP measured only on the CNN; no DP AASIST or SSL run; no ε sweep.
- Model selection on a dev set that reuses training attacks.
- Future: whole-clip scoring, retraining so clean and noisy audio share one
  scale, a frozen-front-end DP SSL-AASIST, a held-out attack set for selection.

## 9. Ethics, privacy and licensing (~500 words)

- Upload privacy: in-memory processing, no logs, video never leaves the
  browser.
- The cost of each error to a real person; why the interface never stamps a
  verdict.
- Dataset licences and attribution; In-the-Wild's evaluation-only use.
- Dual use: a published detector can be used to tune fakes against it.

## 10. Conclusion (~300 words)

## Appendices

- A. Full per-attack tables (`RESULTS.md`).
- B. Every pre-registration and its outcome, verbatim, including both
  overrides.
- C. Reproduction: `hpc/README.md`, job IDs, commit hashes.
- D. Hyperparameters per run.

## References to collect

ASVspoof 2019 database paper and evaluation plan; AASIST (Jung et al. 2022);
SSL-AASIST (Tak et al. 2022); RawBoost (Tak et al. 2022); In-the-Wild (Müller
et al. 2022); XLS-R (Babu et al. 2021); wav2vec 2.0 (Baevski et al. 2020);
DP-SGD (Abadi et al. 2016); Opacus; SpeechFake; People's Speech; LibriSpeech;
Common Voice; VoxPopuli; ASVspoof 5; the TTS systems used in `synth/`.
