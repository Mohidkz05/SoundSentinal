# Brief — Farhan Mohammed

FYP B final paper and presentation, SoundSentinal. Deadline for a first
draft: **[agree a date with Mohid]**.

The paper is LaTeX in the IEEE Transactions template: `paper/main.tex` in the
repo. Build it with `~/tools/tectonic/tectonic paper/main.tex` on Mohid's
machine, or upload the `paper/` folder to Overleaf. Red `[TODO: …]` markers are
the gaps; delete each one as you fill it. **The whole paper must fit in 10
pages**, so your three subsections together get **about 1 page** (≈ 700
words), and each figure fits one column.

You will be asked about your sections in the presentation Q&A, so write what
you understand and can explain.

## 1. Paper: Section II-D to II-F (Background)

This part of the paper is worth **30% of the mark**. The rubric's top band
asks for current *and* seminal papers, and a **critical** review: each
subsection should end with the gap it leaves.

**II-D Generalisation**
- In-the-Wild (`muller2022itw`): detectors trained on ASVspoof collapse on
  real recordings of public figures.
- RawBoost augmentation (`tak2022rawboost`).
- Newer multi-generator datasets: SpeechFake (`speechfake`), MLAAD
  (`muller2024mlaad`), and why we excluded MLAAD (its licence does not allow
  commercial use; `RESULTS.md`, Finding 17).
- **Gap to end on:** papers report EER, so whether a decision threshold
  transfers from one kind of audio to another is rarely tested. Our Findings
  10–16 are about exactly that.

**II-E Differential privacy**
- DP-SGD (`abadi2016dpsgd`), Opacus (`yousefpour2021opacus`), the PRV
  accountant (`gopi2021prv`). What ε (epsilon) means, in one plain sentence.
- **Gap:** no published cost of privacy for spoofing detectors.
- **Critical point:** DP protects the speakers in a *public* training corpus,
  not the user's uploaded clip (`CLAUDE.md`, "Question DP-SGD's premise").

**II-F Deployed detectors** (grey literature)
- What commercial and free detectors show users: a verdict or a single
  accuracy figure. None publishes its threshold or its two error rates
  separately. `DESIGN.md` has a competitor table (Pindrop, Reality Defender,
  Resemble, AI or Not, ElevenLabs); cite their public pages.
- End the section with the research question (already in Section I), and say
  how II-A, II-D and II-F justify it.

## 2. Figures 2 and 3

Both must read in black and white and at one-column width, with labelled axes
and units. Put the image files in `paper/figures/` and add them to
`main.tex` with `\includegraphics[width=\columnwidth]{...}`; write each
caption so the figure makes sense on its own.

- **Fig. 3 — the main figure: In-the-Wild EER by model.** Data:
  `paper/figdata/fig3_itw_eer.csv`. A bar chart in the order given (it tells the
  story: collapse, then recovery). Mark 50% as "chance".
- **Fig. 2 — cost of privacy, per attack.** EER for each attack A07–A19, CNN
  with and without DP. Ask Mohid for the per-attack numbers (they are in the
  evaluation JSON files on M3; summary in `RESULTS.md`, Finding 3). Grouped
  bars, two colours that differ in lightness too.

## 3. Presentation (≈2.5 minutes, you present these)

Monash template: `powerpoint-template-standard.pptx`. For a **general
engineering audience**.

- **Project management** (≈1.5 min): timeline August–October; getting GPU
  time on M3 as the critical path; writing each experiment's success criteria
  down before running it (pre-registration) as a project control; setbacks
  and how they were handled (e.g. the real-world collapse, Finding 6, and the
  missed 2% bar, Finding 18). `RESULTS.md` has dates for everything.
- **Prepared answers** (for the Q&A, 15% of the mark): write short answers
  to the likely questions listed at the end of `WRITEUP.md`, and share them
  with the team.

Keep text on slides short. Agree the hand-over line to the next speaker in
advance.

## Where to look

| For | Read |
| --- | --- |
| The plan for the whole paper and talk | `WRITEUP.md` |
| Every number, with the experiment it came from | `RESULTS.md` |
| Why the models were chosen | `APPROACH.md` |
| Marking criteria | the two FYP B rubric PDFs |
