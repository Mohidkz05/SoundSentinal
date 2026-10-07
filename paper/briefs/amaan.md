# Brief — Amaan Muhammad

FYP B final paper and presentation, SoundSentinal. Deadline for a first
draft: **[agree a date with Mohid]**.

The paper is LaTeX in the IEEE Transactions template: `paper/main.tex` in the
repo. Build it with `~/tools/tectonic/tectonic paper/main.tex` on Mohid's
machine, or upload the `paper/` folder to Overleaf. Red `[TODO: …]` markers are
the gaps; delete each one as you fill it. **The whole paper must fit in 10
pages**, so your three subsections together get **about 1 page** (≈ 700
words).

You will be asked about your sections in the presentation Q&A, so write what
you understand and can explain.

## 1. Paper: Section II-A to II-C (Background)

This part of the paper is worth **30% of the mark**. The rubric's top band
asks for current *and* seminal papers, and a **critical** review: each
subsection should end with the gap or inconsistency it leaves, not just a
summary.

**II-A Benchmarks and metrics**
- The ASVspoof challenges: 2015 → 2019 LA → 2021 → ASVspoof 5. Cite
  `wang2020asvspoof`, `todisco2019asvspoof`, `liu2023asvspoof2021`,
  `wang2024asvspoof5` (keys in `paper/refs.bib`).
- EER (equal error rate) and min t-DCF (`kinnunen2018tdcf`); why t-DCF is the
  primary metric. Plain-English explanation: `RESULTS.md`, "How to read these
  numbers", and `APPROACH.md`, "The comparison table".
- **Gap to end on:** both are pooled, threshold-free numbers, so neither says
  how a detector behaves at the threshold it is deployed at.

**II-B Front-ends** (how audio is turned into features)
- Hand-crafted: CQCC (`todisco2017cqcc`), LFCC (`sahidullah2015lfcc`).
- Log-Mel, and why the Mel scale throws away high-frequency detail where
  vocoder artefacts live (`CLAUDE.md`, "Replace: Mel → LFCC features").
- Raw waveform: SincNet (`ravanelli2018sincnet`).
- Self-supervised: wav2vec 2.0 (`baevski2020wav2vec`), XLS-R (`babu2022xlsr`).

**II-C Back-ends** (the classifier)
- LCNN (`lavrentyeva2019stc`), RawNet2 (`tak2021rawnet2`), RawGAT-ST
  (`tak2021rawgat`), AASIST (`jung2022aasist`), SSL-AASIST (`tak2022ssl`).
- **Inconsistency to point out:** RawNet2's reported LA EER ranges from 0.99%
  to 9.5% depending on the source (`APPROACH.md`, why RawNet2 was dropped).

Read the actual papers for these — the repo files say what *we* concluded, not
what the papers say.

## 2. Verify every reference

`paper/refs.bib` has 27 entries written from memory. For each one, open the
publisher's page (IEEE Xplore, ISCA Archive, ACL Anthology, NeurIPS
proceedings, arXiv) and check authors, title, venue, year and pages. Fix any
that are wrong. Also:
- **Find the SpeechFake paper** and fill in the `speechfake` entry (currently
  a TODO).
- Add any new papers you cite in II-A to II-C, in the same IEEE style.

## 3. Presentation (≈2.5 minutes, you present these)

Monash template: `powerpoint-template-standard.pptx`. The presentation is for
a **general engineering audience**, not a summary of the paper.

- **The problem and why it matters** (≈1.5 min): voice-clone scams, who gets
  hurt, why "98% accurate" claims mislead. Find one or two real, cited
  examples of voice-clone fraud.
- **Constraints** (≈1 min): cost (hosting has used A$0.53 of student
  credit; it scales to zero), carbon (training used 228.7 GPU-hours, roughly
  90–125 kg CO₂e; see `WRITEUP.md`, "Compute used", and check the grid factor),
  privacy (clips are never stored), dataset licences, and the harm of a false
  accusation.

Keep text on slides short; the rubric penalises text-heavy slides. Agree the
hand-over line to the next speaker in advance.

## Where to look

| For | Read |
| --- | --- |
| The plan for the whole paper and talk | `WRITEUP.md` |
| Every number, with the experiment it came from | `RESULTS.md` |
| Why the models were chosen | `APPROACH.md` |
| Marking criteria | the two FYP B rubric PDFs |
