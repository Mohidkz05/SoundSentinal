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

| System | Params | Front-end | EER | Source |
| --- | --- | --- | --- | --- |
| CQCC-GMM | — | CQCC | 9.57% | Official ASVspoof2019 baseline B1 |
| LFCC-GMM | — | LFCC | 8.09% | Official ASVspoof2019 baseline B2 |
| **Our CNN** | 267k | log-Mel | *not run* | — |
| LCNN-LSTM-sum | 276k | LFCC | 1.92% | AASIST Table 2 |
| RawGAT-ST | 437k | raw waveform | 1.19% | AASIST Table 2 |
| AASIST-L | 85k | raw waveform | 0.99% | AASIST Table 2 + official repo |
| **AASIST** | 297k | raw waveform | **0.83%** | AASIST Table 2 + official repo |
| wav2vec2 / WavLM front-end | 95M+ | raw waveform, pretrained | *unverified* | needs VM |

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
