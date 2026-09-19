# SoundSentinal — model and research approach

Decided 17 August 2026. This file records *what we chose and why*, so the
reasoning isn't re-derived every time someone opens the repo. `DESIGN.md` does
the same job for the interface.

---

## The research question

Not "how accurate is our detector" — we would lose that contest to any lab with
a GPU cluster. The question is:

> **What does differential privacy cost an audio deepfake detector?**

Nobody has published that number. Measuring it is the contribution, and it needs
every model trained twice — once normally, once privately — so the gap is
measurable.

This also answers the supervisor's brief (Hui Cui, August 2026): *"please choose
and test the Model first, as we need to show the detection result as the research
component… there are difference models available, you may need to find the one
with the best performance."* She is asking for a benchmarked comparison with a
justified winner, not a single trained network.

---

## The model: AASIST

**AASIST** — *Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
Attention Networks*, Jung et al., ICASSP 2022. Official PyTorch implementation at
[clovaai/aasist](https://github.com/clovaai/aasist).

Two properties matter more than the score:

- **It skips the spectrogram.** It reads the raw waveform through learnable
  band-pass filters (SincNet), so the model decides which frequencies matter.
  Our current log-Mel front-end is actively harmful here: the Mel scale mimics
  human hearing and therefore compresses high frequencies, which is exactly where
  vocoder artefacts live. We are discarding the evidence before the model sees it.
- **It relates distant parts of the clip.** The clip becomes a graph — some nodes
  are frequency bands, some are time segments — and any node can attend to any
  other. A conventional CNN pools, i.e. averages, and a spoofing artefact
  confined to one narrow band during one half-second gets averaged into nothing.
  AASIST's contribution over its predecessor RawGAT-ST is handling frequency and
  time in *one heterogeneous graph* rather than two separate ones.

**Why it beats the alternatives for us specifically:** 297k parameters fits our
8GB card with room to spare, there is official code to clone rather than a paper
to reimplement, and it is still the reference point 2026 papers benchmark
against. It is a manageable step up from a 2-layer CNN; wav2vec2 fine-tuning is
not.

**Scope the word "best" carefully.** AASIST beat every competing single system in
its own 2022 paper. Since then self-supervised front-ends (wav2vec2, WavLM) have
overtaken it, and ASVspoof 5 (2024) is the current benchmark. AASIST is the right
pick for our constraints, not the best model in existence. If asked *"why not
wav2vec2?"* the answer is hardware, not ignorance: those are 95–300M parameters
and need ~24GB of VRAM. See "Hardware" below.

---

## The comparison table

Every row below is on the **ASVspoof2019 LA evaluation partition**, so they are
directly comparable. Rows marked *AASIST Table 2* come from Table 2 of the AASIST
paper; taking them from one source is deliberate.

| System | Params | Front-end | EER | min t-DCF | Source |
| --- | --- | --- | --- | --- | --- |
| CQCC-GMM | — | CQCC | 9.57% | 0.2366 | ASVspoof2019 DB paper, Table 8 |
| LFCC-GMM | — | LFCC | 8.09% | 0.2116 | ASVspoof2019 DB paper, Table 8 |
| **Our CNN** | 267k | log-Mel | **10.15%** | **0.2350** | measured here, 19 Sep 2026 |
| LCNN-LSTM-sum | 276k | LFCC | 1.92% | 0.0525 | AASIST Table 2 |
| RawGAT-ST | 437k | raw waveform | 1.19% | 0.0335 | AASIST Table 2 |
| AASIST-L | 85k | raw waveform | 0.99% | 0.0309 | AASIST Table 2 + official repo |
| **AASIST** | 297k | raw waveform | **0.83%** | **0.0275** | AASIST Table 2 + official repo |
| wav2vec2 / WavLM front-end | 95M+ | raw waveform, pretrained | *unverified* | *unverified* | H100 via `m3h` QOS |

**Two metrics, because the challenge has two.** min t-DCF is ASVspoof2019's
*primary* metric and EER its secondary one. EER scores the countermeasure alone;
t-DCF scores it in the position it actually occupies — in front of a speaker
verification system — and weights each error by what it costs there. A
countermeasure that rejects spoofs the ASV would have rejected anyway has bought
nothing, and EER cannot see that. Normalised, 1.0 is the "accept everything"
floor. `evaluate.py` reports both; the GMM baseline t-DCFs are from the
ASVspoof2019 evaluation plan and the rest from AASIST Table 2.

Why each row earns its place:

- **The two GMM baselines** anchor our numbers to the published literature. They
  are not neural networks — they model what real speech statistically looks like
  and flag outliers. Note they **cannot be trained with DP-SGD at all**, since
  they are not trained by gradient descent. They appear as non-private reference
  points only, and the writeup must say so rather than let a reader assume the
  whole table is private.
- **LCNN-LSTM-sum replaced RawNet2** in this plan. At 276k parameters against our
  267k it is a genuinely controlled comparison — near-identical budget, entirely
  different design, so the difference is architecture alone. RawNet2 was dropped
  because its published LA-eval numbers range from 0.99% to 9.5% across sources
  and no single figure could be defended.
- **AASIST-L** is the fallback if anything runs out of memory. 85k parameters —
  a third the size of our current CNN — and still beats everything except full
  AASIST.

### Licence and attribution — an obligation, not a formality

ASVspoof2019 is released under the **Open Data Commons Attribution License
(ODC-By)**. Free to use, share and modify, including for this project; the one
condition is attribution, and it binds the writeup rather than the code.

> Yamagishi, Junichi; Todisco, Massimiliano; Sahidullah, Md; Delgado, Héctor;
> Wang, Xin; Evans, Nicolas; Kinnunen, Tomi; Lee, Kong Aik; Vestman, Ville;
> Nautsch, Andreas. (2019). *ASVspoof 2019: The 3rd Automatic Speaker
> Verification Spoofing and Countermeasures Challenge database*, [sound].
> University of Edinburgh. The Centre for Speech Technology Research (CSTR).
> https://doi.org/10.7488/ds/2555

Note what ODC-By covers: the *database*, not necessarily each recording in it.
The corpus is built on VCTK, whose speakers consented to research use — so
redistribution of the audio is not ours to grant. Cite it, do not re-host it,
and keep the corpus on M3 and the laptop rather than in the repo (`data/` is
already gitignored, which is the mechanism enforcing this).

### What the baseline actually measured — 19 September 2026

The CNN row above is now a measurement. Non-private, 5 epochs, class-weighted,
LA eval partition. Three things came out of it, and the second is the one to
lead with.

**1. It does not beat the official baselines.** 10.15% eval EER sits *above*
both LFCC-GMM (8.09%) and CQCC-GMM (9.57%) — worse than two systems from 2019
that are not neural networks at all. That is not a disappointment, it is the
argument for this project: it shows a small log-Mel CNN is the wrong tool, and
it is the measured gap that AASIST at 0.83% is proposed to close. A comparison
whose starting point already worked would not be worth running.

**2. Dev EER was 0.24%. Eval EER is 10.15%. That is a factor of 42.** The dev
partition reuses the six attacks seen in training, so 0.24% measured how well
the model recognised six specific vocoders, not whether it detects synthetic
speech. Anyone reporting the dev figure would be claiming to beat AASIST with a
two-layer CNN. This is the concrete instance of the warning below, and it is
worth quoting in the writeup as a methodology point rather than hiding.

**3. The calibrated threshold does not survive the partition change**, and this
one has product consequences. The dev-calibrated operating point is 0.5698; on
eval the EER point is **0.0070**, nearly two orders of magnitude lower. Serving
the dev threshold against unseen attacks gives:

| | at eval's own threshold | at the served threshold 0.5698 |
| --- | --- | --- |
| accuracy | — | 70.78% |
| spoofs passed | 10.15% | **32.51%** |
| real clips flagged | 10.15% | 0.61% |

A third of deepfakes get through. The model is not the only thing at fault —
the threshold is, and it is a threshold this product *draws on screen* and
describes in prose on `/result`. Calibrating on a partition that shares attacks
with training produces an operating point that is confidently wrong in
deployment. Whatever architecture wins, calibration needs a held-out set whose
attacks are unseen.

**Per-attack, the spread is enormous**: A07 0.11%, A16 0.22%, A09 0.63% at one
end; **A17 41.19%**, A13 15.85%, A18 12.19% at the other. A17 being close to
useless is consistent with the wider literature, where it is routinely the
hardest LA attack. The pooled 10.15% is therefore an average over attacks the
model handles completely differently, and a writeup that quotes only the pooled
figure describes a detector that does not exist. Report the breakdown.

Raw numbers are in `checkpoints/nodp/eval_eval_*.json`, written by
`evaluate.py`, so the table above can be rebuilt without rerunning anything.

### Per-attack, against the official baselines

The pooled numbers put our CNN between the two GMM baselines — worse than both
on EER, marginally better than CQCC-GMM on min t-DCF. That the two metrics rank
the systems differently is itself the argument for reporting both.

Per attack, though, the story is not "slightly worse". It is "wildly uneven".
Baseline columns are B1/B2 from Table 8 (evaluation set) of the ASVspoof2019
database paper — the same table the pooled row comes from, so the comparison is
like for like.

| Attack | Ours | B1 CQCC-GMM | B2 LFCC-GMM | |
| --- | --- | --- | --- | --- |
| A07 | 0.11% | 0.00% | 12.86% | |
| A08 | 0.83% | 0.04% | 0.37% | |
| A09 | 0.63% | 0.14% | 0.00% | |
| A10 | **3.70%** | 15.16% | 18.97% | we beat both, by 4–5× |
| A11 | 2.97% | 0.08% | 0.12% | |
| A12 | 8.10% | 4.74% | 4.92% | |
| A13 | 15.85% | 26.15% | 9.57% | |
| A14 | **2.12%** | 10.85% | 1.22% | |
| A15 | 5.72% | 1.26% | 2.22% | |
| A16 | 0.22% | 0.00% | 6.31% | |
| A17 | **41.19%** | 19.62% | 7.71% | we are 5× worse than B2 |
| A18 | 12.19% | 3.81% | 3.58% | |
| A19 | **2.61%** | 0.04% | 13.94% | |
| **Pooled** | **10.15%** | 9.57% | 8.09% | |

Two things to take from this.

**Our CNN is not uniformly inferior — it is differently shaped.** On A10 it beats
both baselines by a factor of four to five, and on A19 it beats LFCC-GMM by five.
Those are neural waveform-generation attacks, and a learned front-end sees
something the cepstral ones do not. The pooled figure hides that entirely.

**A17 is the whole deficit.** At 41.19% the model is close to useless on it,
against 7.71% for LFCC-GMM. Remove A17 and our pooled EER would sit comfortably
below both baselines. The database paper notes A17 is VC with waveform
filtering and that its waveform generation method is unlike anything in the
training set — so this is a generalisation failure on a specific synthesis
family, not a uniformly weak detector. It is also a known-hard attack: B1 scores
19.62% on it.

That reframes the case for AASIST. The argument is not "our model is bad"; it is
that a log-Mel front-end throws away the evidence for one attack family, exactly
as predicted in "The model: AASIST" above — the Mel scale compresses high
frequencies, which is where waveform-filtering artefacts live. A17 is that
prediction showing up as a number.

### Sourcing discipline

**The same model name scores wildly differently across papers.** LFCC-GMM is
quoted anywhere from 8% to 21%. That is not sloppiness: ASVspoof2019 LA,
ASVspoof2021 LA and ASVspoof 5 are different test sets. Never mix rows from
different papers into one table without confirming the evaluation partition
matches. Prefer re-running a system ourselves over quoting it.

Also: **report eval-partition EER, not dev.** The dev partition uses attacks
A01–A06, the same ones seen in training. Eval has unseen A07–A19. Dev EER is
optimistic.

---

## Differential privacy: off now, ready later

**Decision: train without DP for the main results, but keep every architecture
DP-compatible.**

The reasoning:

- DP-SGD protects the privacy of the **training data**. Our training data is
  ASVspoof2019 — a publicly released corpus. We would be paying accuracy to
  protect speakers whose audio is already published.
- DP does **nothing** for the privacy a user of this tool actually cares about —
  the clip they upload. That is an inference-time concern, and the fix is
  operational: never write the upload to disk, process in memory, delete
  immediately, don't log it. Cheap, and it must be stated separately in the
  writeup so the two guarantees aren't conflated.
- DP-SGD is **not a switch**. It changes how the weights are computed, so it
  cannot be applied to an already-trained model. Any run over sensitive data
  without it is permanently compromised.

**Staying DP-ready costs one constraint: no BatchNorm, anywhere.** BatchNorm
mixes statistics across a batch, which breaks DP-SGD's per-sample gradient
guarantee, and Opacus refuses to wrap such a model. AASIST ships with BatchNorm,
so porting it means substituting **GroupNorm**. That is contained work, and
"adapting AASIST for differentially private training" is itself a legitimate
contribution.

**When DP switches back on:** the day we train on user-contributed voice data.
The correct pattern then is pretrain on public data with DP off, then fine-tune
on client audio with DP on — the sensitive data only ever touches the private
phase. `train_dp_avspoof.py` already implements both paths; `--no-dp` is what
*skips* Opacus, so re-enabling it is omitting a flag. Expect to re-tune noise
multiplier, batch size and learning rate. DP is a technical guarantee, not a
legal one — training on client voices needs their consent regardless.

---

## Order of work

1. **Download ASVspoof2019 LA.** ~7–10GB, peaking near 20GB while zip and
   extracted copy coexist. This is the genuinely slow step.
2. **Non-private baseline on the current CNN** (`--no-dp`). ~10 min on CPU. This
   is the floor and the first honest number to show the supervisor.
3. **Port AASIST**, BatchNorm → GroupNorm. Train non-private.
4. **Fill the comparison table** — LCNN-LSTM-sum and AASIST-L as time allows.
5. **DP arm** on whichever architectures fit the timetable, for the cost-of-
   privacy measurement.
6. **Evaluate on In-the-Wild** (Müller et al.) with no retraining. LA's attacks
   predate modern voice cloning, so the gap between LA-eval and In-the-Wild is a
   finding in itself and costs one evaluation pass.

Log a confusion matrix throughout, not just accuracy and EER. On 1:9 data
"90% accurate" can mean "always guesses spoof".

---

## Hardware

Local: RTX 4070 **Laptop, 8GB**, 16 cores, WSL2 with 6.7GB RAM, ~300GB free.
Everything in the table above except the wav2vec2 row fits comfortably.

**Granted, and it is better than the VM that was asked for.** The request for a
24GB GPU VM was answered with a Monash **M3 (MASSIVE)** account instead —
project `df37`, *Detecting Deepfakes Without Compromising User Privacy*, active
14 September 2026, username `mkha0155`. That covers the wav2vec2 / WavLM ceiling
row with far more headroom than the 24GB floor we asked for, and there is no
machine to maintain.

It is a **Slurm cluster, not a VM**, and the difference is not cosmetic: no
sudo, no long-lived interactive session, jobs submitted against
`--account=df37` and run when a GPU frees up. `hpc/README.md` is the runbook and
`hpc/` holds the scripts. Nothing is blocked on it — the CNN baseline still
runs locally in ~10 minutes — but the big-model rows now have somewhere to go.
