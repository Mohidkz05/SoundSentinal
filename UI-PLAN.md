# UI refinement plan — 1 October 2026

A refinement, not a redesign. The identity in `DESIGN.md` stays: an instrument,
not a verdict machine; Sentinel Teal for chrome only; the violet↔orange verdict
scale for readings only; the threshold drawn on screen. The job is to make what
exists uniform, correct and finished.

Evidence: screenshots of every route at 1440 and 390 px, light and dark,
through the real upload path with the served model; the impeccable detector
(one finding — the needle overshoot easing, a documented exception, kept);
competitor review below.

## What competitors do, and what we take from it

| Product | Pattern | Take / leave |
| --- | --- | --- |
| Resemble Detect | "Most tools return a score. We return a verdict and an explanation"; heatmaps of *where* | Take: every number explained in place. Leave: the verdict. |
| AI or Not | One drop zone, clear idle/working/done states, accuracy stats as trust | Take: explicit states. Leave: unmeasured "98.9% accuracy". |
| Reality Defender | Laddered CTAs, trust by third-party validation | Take: one clear primary action per screen. |
| Poynter's test of four detectors (2024) | Failures: unexplained %s, binary verdicts that contradict each other, error states with no next step, **no disclosed error rates** | Our thesis already answers all four. Make that visible on the home page: measured error rates, live from the served model. |

## Findings, by severity

**P0 — brand or layout broken**
1. **Archivo and IBM Plex Mono never render.** The font variables are set on
   `<body>`, but `--font-sans` is resolved on `<html>`, where they don't exist,
   so every page has shipped in the system sans.
2. **The display heading overflows at 390 px** ("instrumen|t"): `--text-display`
   is a fixed 68 px.

**P1 — uniformity and correctness**
3. `<Link><Button>` puts a `<button>` inside an `<a>` (invalid interactive
   nesting, two tab stops) on every CTA.
4. Spacing has no scale: sections use py-10/12/14, pt-14/16/20/24, mt-2…14 ad hoc.
5. Touch targets under 44 px: theme toggle (36), nav links (~36), `sm` buttons (36).
6. Two copies each of the stat readout and the data table, styled separately.
7. Eyebrow labels above headings ("Audio authenticity analysis", "Reading" …):
   decoration posing as structure. "Step 1 of 2" carries real information and
   becomes a stepper.
8. Navigation lists "Result" (a step, empty when visited directly) and "Design"
   (a reference for builders) beside the product.
9. No footer: every page ends differently; the research disclaimer floats.
10. Factual copy is out of date: /upload says the served model was trained with
    differential privacy (it wasn't — DP was measured on the CNN only) and "on
    one corpus of one kind of attack" (it is LA + SpeechFake). /result's
    "Unseen attacks" predates SpeechFake.

**P2 — finish**
11. Upload status line ("48 KB · READY") sits over the intake bars and loses contrast.
12. Home's secondary CTA goes to the design system rather than explaining the product.
13. The result page's only action is at the very bottom.

## Tasks, in order

- [x] **T1 Fonts** — font variable classes on `<html>`; verify computed family.
- [x] **T2 Fluid type** — display/h1/h2/readout scale with `clamp()`; no overflow at 360 px.
- [x] **T3 Spacing scale** — `--space-*` tokens (4 px base) and semantic roles:
      `section` rhythm, `stack` gaps, heading→content distance. Replace ad-hoc values.
- [x] **T4 Touch targets** — 44 px minimum hit area for every control.
- [x] **T5 Button as link** — `Button` renders a Next `Link` when given `href`; remove nesting.
- [x] **T6 Shared components** — `SectionHead` (heading + lede pair), `Stat`,
      `data-table` styles, `Notice` (error / caveat), `Stepper`, `Footer`.
- [x] **T7 Header & IA** — nav: Analyse · How it works · (Design → footer);
      Result reachable from the flow only; wordmark at every width.
- [x] **T8 Home** — no eyebrows; CTA pair (Analyse a clip / How a reading works);
      "Measured, not claimed" strip fed live from the served model; disclaimer to footer.
- [x] **T9 Upload** — stepper, legible status chip, factual copy fixed.
- [x] **T10 Result** — stepper, actions near the reading, factual copy fixed,
      shared stat/table/notice components.
- [x] **T11 /design** — document the spacing scale, sizes, new components.
- [x] **T12 DESIGN.md** — spacing, components, navigation, footer, the font fix.
- [x] **T13 Verify** — one batched screenshot round (desktop + mobile, both
      themes), keyboard path, contrast of new pairs, detector, `next build`;
      fix in one batch, confirm once.

## Outcome

All thirteen tasks done. Verified in one batched round (4 routes × 2 themes ×
1440/390 px, real upload through the served model) plus one confirmation round
after fixing what it showed: the error-rate table scrolling sideways on phones
(now stacked blocks) and the upload's "change file" control losing contrast on
the intake surface (now inside the file chip). Keyboard path: every stop ≥44px
with a visible ring; the hidden file input was removed from the tab order.
Detector: one finding, the needle's overshoot easing — a documented exception.
