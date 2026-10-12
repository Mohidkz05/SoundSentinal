# SoundSentinal — design system

**Live reference: `/design`** (under `npm run dev` only; production builds serve it as a 404 — `src/app/design/layout.js`). It renders from the same CSS the product does, so
it cannot drift. Read it before building a screen.

Everything is defined in `src/app/globals.css`. Nothing here is decorative — if
you can't find a token for what you need, add it there first rather than writing
a one-off value in a component.

---

## The thesis

**An instrument, not a verdict machine.**

The detector returns a score, not a fact. The served model is good — 2.65% EER
on real-world audio — but at its threshold it still flags 0.6% of genuine
recordings, misses 3.6% of real-world fakes and nearly half of clean studio
fakes (RESULTS.md Finding 12). A UI that stamps a red **FAKE** on a clip would
be claiming a confidence the model does not have.

Three consequences run through every decision below:

1. The brand colour never renders a result.
2. The result scale loses its colour where the model is least certain.
3. The decision threshold is drawn on screen rather than hidden behind a label.

If a proposed change breaks one of these, it's the wrong change.

---

## Colour

**Redesigned 10 October 2026** ("Bright instrument", the owner's choice after a
competitor review): the old teal-on-pale-petrol world read as washed out. The
rules below are unchanged; the colours are new.

Three families. **A colour does one job**; the moment the brand indigo appears
in a result, or a verdict orange appears on a button, the system stops meaning
anything.

### Sentinel Indigo — brand, actions, focus, colour bands

Hue 280° in OKLCH, a vivid blue-violet: `--color-indigo-50` … `-950`,
`--accent` = indigo-700 `#4b2bd7` (8.0:1 on white). In dark mode the accent is
a pale periwinkle `#c6ccff`, because the saturated indigo fails 4.5:1 on
near-black and collapses onto the azure verdict end for protanopes (ΔE 0.018).

Use the semantic aliases: `--accent`, `--accent-hover`, `--accent-quiet`,
`--accent-wash`, `--accent-contrast`.

**Colour bands.** A long page is structured by full-bleed brand surfaces, not
only hairlines: `surface-brand` (indigo-700) and `surface-brand-deep`
(indigo-900, the footer). Both *redefine the semantic tokens inside
themselves* — text, lines, accent, focus ring — so anything placed in a band
re-themes with no per-component variant. Secondary text on a band is a tint of
its hue (indigo-100 on indigo-700, 6.6:1), never grey. One band per page body
at most; on `/` it is "How it works", on `/result` "What it can't tell you".

The field, measured 10 October: Resemble is a deep rust gradient, Pindrop
orange on brown, Hiya violet, Sensity indigo with warm yellow, AI or Not a dark
HUD. The vivid ones commit to one bold field of colour and show the product in
the hero; that is what this palette does.

### Ink — surfaces and text

The same 280° hue at low chroma (`--color-ink-*`). Neutrals that sit under the
brand rather than fighting it; the dark theme is an indigo-black, not grey.

Use `--canvas`, `--panel`, `--raised`, `--overlay`, `--line`, `--line-strong`
and `--text-primary/secondary/muted/faint`.

### Verdict scale — results only

Diverging **azure ↔ orange**, the canonical colourblind-safe blue/orange axis.
It was violet ↔ orange until the brand became indigo-violet; it moved so a
reading can never be mistaken for chrome.

- **Hue** says which side of the decision threshold the reading falls on.
- **Chroma** says how far from it. The scale drains towards grey at the
  boundary, so "the model is unsure" is expressed by the colour draining out —
  not by a fourth hue nobody can name.

Chosen on measurements, not taste: simulated through protanopia, deuteranopia
and tritanopia (Machado 2009), scored on OKLab ΔE, worst case over all four
kinds of vision:

| Check | Light | Dark | Bar |
| --- | --- | --- | --- |
| Across the threshold (possibly ↔ likely) | 0.097 | 0.119 | 0.096 |
| Endpoints (unlikely ↔ very likely) | 0.226 | 0.227 | 0.207 |
| Brand accent vs either end | 0.119 | 0.18 | 0.10 |

The bars are the old palette's own measured values, so the redesign is no
worse for colourblind readers than what it replaced. The earlier finding still
holds: a brand-aligned verdict (teal ↔ ember, ΔE 0.048 under protanopia) fails,
which is why **the brand colour cannot double as the verdict colour**.

Every verdict colour clears 4.5:1 on its own surface in both themes.

#### The ramp is never drawn as a bar, and never drawn on the track

`verdict-ramp` is the continuous diverging colour, and **only `/design` renders
it raw**, as documentation of the palette. A reading is drawn by
`components/ui/verdict-scale.js`, which is an engraved **neutral** scale —
`graduation-fine` + `graduation-major` masks over `bg-line-strong` /
`bg-muted` — with hue carried by the needle, the readout and the tier name.

Two separate decisions, both arrived at by removing something.

**Not a filled bar.** A continuous fill reads as a *progress bar*, a quantity
accumulating towards completion, and a probability is not that. It also drags the
eye along its whole length when the only thing carrying the result is one point
on it. The graduated form says "read a position off me", which is what the number
actually is: fifty fine ticks, one per half point of the model's score, with a
full-height major every 2.5.

**The axis is the model's score (log-odds), not a percentage.** The SSL model
pushes P(spoof) to within a hair of 0 or 1 and its threshold is P = 0.99966, so
on a 0–100% axis the threshold and every high reading shared the last pixel, and
fixed probability tiers headlined real clips "Very likely AI generated". The
scale runs from −11.25 to +13.75, chosen so the existing masks put majors on
multiples of 2.5 and numerals can sit on −10, −5, 0, +5, +10.

**The headline is a confidence, not a probability** (4 October 2026, owner's
request). The big readout is `confidence()` from `src/lib/verdict.js`:
sigmoid(|score − threshold|), so the threshold reads 50% and each unit past it
multiplies the odds by e. It is labelled with the side it supports ("Model
confidence it is real") and carried by a note saying it is certainty, not
accuracy, pointing at the measured error rates. The score moved to the line
under the headline. The axis stays in log-odds; the percentage never draws it.

**Not a coloured track.** A coloured track states a verdict at every point on the
axis, including all the points the model said nothing about. On a real instrument
the face is neutral and the *pointer* carries the state.

Both colour rules above survive intact, because they were always rules about the
reading rather than about the track: hue still says which side of the threshold,
and the middle tiers are the desaturated tokens, so a needle near the line still
drains towards grey.

Mask geometry: both masks put the tick at 0.6% of the band's width and **centre
it in its cell**, which lands fine ticks on the odd percentages and majors on
5, 15 … 95. So every fifth fine tick is a major, the band is inset equally at
both ends instead of running out of ticks on the right, and the threshold
annotation falls in a gap rather than on top of a graduation.

The scale is shared by the calibration meter, the home page illustration and
`/design`. Don't hand-roll a second copy.

### Danger — system failures only

Hue 25°, held 20° off the verdict orange, so "the upload was rejected" never
reads as "this audio is fake". Use it for errors and destructive actions. Never
for a result.

---

## Type

**Changed 10 October 2026: Bricolage Grotesque + Martian Mono** replaced
Archivo + IBM Plex Mono, which the owner found generic (and Plex Mono is the
default "technical" mono of half the developer tools on the web).

**Bricolage Grotesque** (OFL) for everything that is words. Its optical-size
axis does the job the width axis used to: at headline sizes the browser draws
the display cut — ink traps, tight apertures, a voice with some grit, which
suits a tool about telling a human voice from a synthetic one — and at
15–17px the same file draws calmer, opener text shapes. One 131 KB variable
file covers 200–800 weight, 75–100 width and 12–96 opsz. Headings are 700 and
tracked looser than Archivo was (−0.012em display): the display cut is
already tight, and −0.03em made "it's" touch.

**Martian Mono** (OFL, Evil Martians) for every number, with tabular figures,
because every number this app shows is a measurement: probabilities,
thresholds, sample rates, durations. It is an engineered, squared mono — an
instrument panel's numerals rather than a code editor's — and its width axis
sets the wordmark at 112.5, so the brand nameplate and the readouts share one
voice. 38 KB, variable.

**Rejected:** Inter, Space Grotesk, JetBrains Mono, Fira (the generic AI
defaults; the font-pairing lookup returned exactly these); Funnel Display
(characterful but weak at body size); a serif (no job here the optical-size
axis doesn't already do).

Use the generated utilities — `text-display`, `text-h1`, `text-h2`, `text-h3`,
`text-body`, `text-small`, `text-label`, `text-readout`, `text-score`,
`text-tick`. `display`, `h1`, `h2`, `h3` and `score` are fluid (`clamp()`
between a 360px phone and a 1440px desktop); body sizes are fixed. A fixed
68px display once overflowed a phone.

### The roles (type audit, 10 October 2026)

| Role | Style | Where | Why |
| --- | --- | --- | --- |
| Display | Bricolage 700 (display opsz), 43→68px | page headline | the one voice; expanded width does what a second family would |
| Score | Martian Mono 500, 75→144px | the reading on `/result` | the product is the number; tabular so it never reflows |
| Section heading | Bricolage 700, 28→32px | every band | |
| Step / card title | Bricolage 700, 18→22px | steps, tips card | was a fixed 22px: on a phone it sat at 1.1× under a 24px h2 and read as the same rank. Now ≥1.45× apart at every width |
| Lead | Bricolage 400, 19px | the hero sentence, the wordmark | |
| Body | Bricolage 400, 17px, lh 1.6 | prose, nav links, **section intros and step text** | raised from 16px for the general public reading on a phone; intros and steps were 14px |
| Small | Bricolage 400, 15px, lh 1.5 | captions, footnotes, metadata | was 14px |
| Field label | Bricolage 600, 14px, muted, sentence case (`field-label`) | "Duration", "Real flagged", table headers, "Model confidence…" | these were 11px mono caps: below the 12px floor, and mono on words that are not measurements is costume. Weight and tone separate a label from 14px body; a costume does not |
| Scale marking | Martian Mono 400, 12px caps, +0.12em (`tick-label`) | tick numbers, threshold mark, band names, the score line | the only place mono caps are earned: it is a measuring instrument's engraving. Up from 11px |
| Readout value | Martian Mono, 18→22px | error rates, clip facts | measurements, tabular |

Buttons use `text-body` (md, lg) and `text-small` (sm); they used three
off-scale sizes (13, 15, 16px). **The header** is 72px tall with a 36px mark, an 18px wordmark (hidden below 640px, where mark + wordmark + nav + toggle need ~500px) and 17px nav links.

**WCAG check, 10 October 2026**, at 1440 / 720 (200% zoom) / 390 / 320px, light and dark: no horizontal scroll at 320px (1.4.10), no clipping under the 1.4.12 text-spacing override, every text element ≥ 4.5:1 (≥ 3:1 large) against its actual background (1.4.3), every control ≥ 24px (2.5.8). Smallest visible text: 14px; only the scale's engraving is 12px, and it is `aria-hidden` with the reading stated in text.

Prose is held to 45–75 characters: the
`Details` panels were running 97 characters a line and are now capped at
68ch of their own size. Nothing in the product renders below 12px.

**Font weights in use:** Bricolage 400/500/600/700; Martian 400/500/600. Both are single variable files, so weights cost nothing extra.

**The font variables live on `<html>`.** `--font-sans` is declared in
`@theme`, which Tailwind emits on `:root`, and a custom property resolves its
`var()` where it is declared. With `archivo.variable` on `<body>`, the brand
font was undefined at `:root` and every page rendered in the system sans until
1 October 2026 — check the computed `font-family` if type ever looks generic. They carry the paired
line-height and letter-spacing. **Do not write `text-[length:var(--text-h1)]`** —
the arbitrary-value form sets font-size only and silently drops both.

---

## Surface, rounding, elevation

`panel` and `panel-raised` are the two containers. Both carry a hairline border
and a 1px top edge highlight — the single device that makes the UI read as
instrument rather than website.

Elevation is theme-specific by necessity. On paper it's an ordinary shadow
tinted with the petrol hue. On the dark panel it's surface lightness plus the
edge highlight, because a large black shadow on a dark ground just reads as mud.
Same token name, defined twice.

Radii: `tick` 2 · `xs` 4 · `sm` 6 · `md` 10 (buttons, inputs) · `lg` 14 (panels)
· `xl` 20 (the meter housing, modals). Nothing rounds past 20 except pills.

---

## Layout

**Full bleed.** There is deliberately no max-width: the app is an instrument
panel, and an instrument uses the whole face of the case. The only horizontal
constraint is `--gutter` (`clamp(1.25rem, 3vw, 3.5rem)`), applied by the `shell`
utility, which the header and every page use — so the header's bottom rule lines
up with the content edges. `edge` is the same gutter without block padding, and
`rule-full` is a hairline that spans the bleed. At this width **a rule is the
main structural device**, doing the job a container edge used to.

**Full bleed must never mean a 200-character line.** Nothing inside runs the full
width by default: prose keeps its own `ch`-based measure. What earns the width is
content where width does real work:

- The **verdict scale**, because on a graduated scale width *is* resolution:
  a longer scale puts more distance between two readings a tenth of a point
  apart.
- The **waveform**, because it is a signal on a time axis and width buys
  visible detail.
- The **intake deck** on `/upload`, which is both the illustration and the
  drop target and so is the page. Since 10 October it is a framed `raised`
  surface (radius-xl) with the way in — drop, *Choose a file*, *Record* —
  laid over the faded idle timeline (below it on phones). The whole window
  takes a drop; dragging a file over the page draws a dashed accent outline
  round the deck, and over the deck it turns solid with the accent wash. The
  dashed outline appears only while a drag is real — never as resting
  decoration. Recording draws the take onto the ruler live (the lit window
  fills first) with its controls in a strip under the ruler, never on top of
  the take. The browser's echo cancellation, noise suppression and gain are
  off: they are processing, which is what the model looks for.

Those three span the bleed. Everything else is a grid.

**A heading and its paragraph go side by side, not stacked.** A heading with a
measure-limited paragraph under it leaves most of a 1700px row empty, because the
paragraph is constrained and the heading is not. Pair them across the row
(`lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]`) and the width is used without a
line of text growing past reading length.

Where a section has several short pieces of prose, run them as columns
(`lg:grid-cols-3`) rather than one narrow stack.

A rail must never be an empty column waiting for data. Carry the standing
explanation in it at all times and slot measurements in above them.

One caveat on `ch` units: `max-w-[Nch]` on a display-sized heading resolves
against the *heading's* font size, not the body's, so it comes out far narrower
than it looks. Size hero text columns in rem.

---

### /upload is one instrument panel (10 October 2026)

Headline and sentence above at full width; below them **one raised panel**
that is the whole drop target: the controls centred at the top, the intake
field in its own strip under them, the three tips along its foot. It replaced
a full-bleed field behind the page with the copy squeezed into a column and a
tips card floating at the right — three things that read as unrelated, with
the controls set over bars. Nothing is read over the field now, so it needs no
wash; a drag over the panel draws an accent ring round it.

## Space

One 4px-based scale, **named by role**, in `globals.css`:

| Token | Size | Separates |
| --- | --- | --- |
| `--space-tight` | 0.375rem | a label from the thing it labels |
| `--space-stack` | 0.75rem | items inside one group |
| `--space-group` | 1.75–2.5rem | sibling groups inside a band |
| `--space-head` | 2–2.75rem | a band's heading block from its content |
| `--space-section` | 3.5–6rem | block padding of a band |

Pages below their first screen are built from **`band`** — a full-bleed rule,
the gutter and `--space-section` — so the distance between sections is the same
on every page. Use the tokens (`mt-[var(--space-head)]`), not ad-hoc `py-12` /
`mt-10`; that drift is what this scale replaced.

## Controls

Four button variants in `components/ui/button.js`: `primary`, `secondary`,
`quiet`, `danger`. Three sizes drawn at 36 / 44 / 52px.

**One `primary` per screen.** If two things are primary, neither is. Default to
`quiet` for anything that isn't the point of the screen.

**`<Button href>` renders a link.** Navigation is a link, an action is a
button. Never wrap a `<Button>` in a `<Link>` — that is a button inside an
anchor: invalid nesting and two tab stops.

**Every control has a 44px hit area** (`--hit`), whatever it is drawn at. `sm`
buttons extend their target with an invisible `::after`; nav links, the theme
toggle and stepper links set `min-h-[var(--hit)]`.

Press feedback is a 1px downward nudge, never a scale — text stays on the pixel
grid instead of softening mid-press.

## Shared components

Each pattern exists once, in `components/ui/`:

- `SectionHead` — every band heading: title left (24rem), explanation right
  at `lg`, stacked below it.
- `Stat` — one measured value (tick label, tabular value, note).
- `Notice` — `caveat` (muted, qualifies a result) or `danger` (`role=alert`,
  the danger token). Icon plus text, never colour alone.
- `Stepper` — the two-step analyse flow; an ordered list with
  `aria-current="step"`.
- `ErrorRates` — measured error rates per test set; a table from `sm`, stacked
  blocks on phones. Fed only from the API, never typed in.
- `data-table` (utility) — every table's hairlines, header type and padding.

**No eyebrows.** A tick label above a heading is decoration posing as
structure; the heading carries its own weight. Tick labels label values,
scales and table columns.

## Navigation and page frame

Header nav is the product, not a site map: **Analyse** and **How it works**.
`/result` is reached through the flow (the stepper links back to it); `/design`
is for builders and lives in the footer. Every page ends with `Footer`, which
carries the standing disclaimer and the training data's licence credits.

---

## Components: shadcn/ui on this system's tokens (10 October 2026)

Interactive primitives come from **shadcn/ui** (Radix underneath), copied into
`components/ui/*.jsx` and **re-themed**: every shadcn token name was mapped to
ours (`bg-primary`→`bg-accent`, `text-muted-foreground`→`text-muted`,
`bg-background`/`bg-card`→`bg-raised`, `border-input`→`border-line-strong`,
`destructive`→`danger`…), and shadcn's own focus ring was removed in favour of
the global `:focus-visible` outline. Its default zinc look is the v0 / AI
default, so it is never used as shipped.

| Piece | shadcn component | Why |
| --- | --- | --- |
| `Button` (`button.js`) | Button (cva + Slot) | variants as before; links render as `<a>` through Slot |
| Sample toggle | ToggleGroup | one tab stop, arrow keys; Motion `layoutId` highlight slides |
| `/result` Details | Accordion | animated open/close; several open at once |
| `Notice` | Alert | `danger` boxed, `caveat` inline (it sits inside cards) |
| Error-rate table | Table | row hover tint |
| Theme toggle | Tooltip | names the action |

**Do not run `shadcn init`, and back up `globals.css` before `shadcn add`**:
both rewrite it with shadcn's zinc variables (it happened once and was
restored). `components.json` is hand-written; the `@components/*` alias in
`jsconfig.json` is what lets the CLI find `components/ui`. `cn()` in
`src/lib/utils.js` registers this system's text sizes with tailwind-merge, or
`text-body` and `text-secondary` would be merged as one group and one dropped.

**Accents and shadows.** The primary button sits on a brand-tinted shadow
(`--shadow-action`) and lifts 1px on hover; in dark mode it glows instead of
casting. Step numbers on the brand band are solid white discs. Static panels
do not lift on hover — a lift says "clickable".

## Motion

**Framer Motion (`motion/react`) is used through `components/motion/reveal.js`**
(10 October 2026): `Rise` above the fold (travels, starts opaque — LCP),
`Reveal` below it (fades and rises once, at 20% in view), `Stagger` /
`StaggerItem` for lists whose order is the information. Plus: the sample
toggle's sliding highlight, the upload timeline unrolling a chosen clip left
to right (3ms per bar) and its end marker. `MotionConfig reducedMotion="user"`
in `components/providers.js` strips travel for reduced-motion users.

Tokens live in both `globals.css` and `src/lib/motion.js`, and the two must stay
in agreement. A button that hovers in CSS and mounts in Motion should feel like
one product.

| Token | Value | Use |
| --- | --- | --- |
| `--duration-tap` | 100ms | press states |
| `--duration-fast` | 160ms | hover, focus |
| `--duration-base` | 240ms | panels, disclosure |
| `--duration-slow` | 420ms | page entrance |
| `--duration-sweep` | 900ms | needle travel |
| `--ease-instrument` | `0.2, 0.8, 0.2, 1` | everything |
| `--ease-needle` | `0.34, 1.28, 0.44, 1` | the reading, and nothing else |

`--ease-needle` overshoots ~4% and settles, the way a real meter does. It is
reserved for the reading. Use it anywhere else and the gesture is spent — it
stops meaning "a measurement landed".

`prefers-reduced-motion` is honoured globally in `globals.css`.

---

## The signature: the calibration meter

`components/ui/calibration-meter.js`. The element the product is remembered by.

Every competitor renders a verdict. This renders a **reading**, against the
model's own published decision threshold. No competitor shows their cutoff;
drawing it turns an unexplained verdict into a number you can argue with, and it
consumes the `threshold` the trainer already computes from dev-set EER and
currently throws away.

Three channels carry the result — needle position, tier name, numeric readout —
so colour is never load-bearing on its own (WCAG 1.4.1).

**The tiers are placed around the threshold.** Clear, below the uncertain band;
Uncertain, from the band's lower edge to the threshold (not flagged, but where
genuine speech sometimes scores); Flagged, from the threshold up by the band's
width; Strong, beyond. The band's lower edge is the 95th percentile of held-out
real speech, read from the checkpoint by `app.py`; the upper edge mirrors it and
is a convention, which `/result` states. The uncertain band is drawn as a
neutral bracket under the engraving — a region of the axis, not a reading. See
`src/lib/verdict.js`.

**The measured error rates are shown on `/result`** ("How often it is wrong"),
from `evaluate.py` reports placed beside the checkpoint as
`best.measured-<name>.json`. `app.py` drops any report measured at a different
threshold, so the page cannot show rates for an operating point it is not
using.

---

## The home page hook: a sample reading

Since 6 October 2026 the hero is a question in the visitor's words ("Heard a
voice. Not sure it's human?") beside `components/ui/sample-reading.js`: the two
bundled clips as the live model actually scored them, cycling until the visitor
picks one. The playhead scans the waveform, then the needle travels, so a swap
swings it across the threshold. It is DOM and CSS, so it renders without WebGL;
the hero previously depended on the spectral field and was half empty without
hardware acceleration.

Its numbers come from `src/lib/samples.json`, written by
`ai_model/sample_readings.py` against the served API. **Rerun it whenever the
checkpoint changes.** The clips are from the model's training data, and the
caption says so: the panel shows what a reading looks like, not accuracy.
The home page is otherwise three steps and the live error rates; the
design-rule explanations live on `/design`.

---

## The 3D layer

Six WebGL scenes, in `components/three/`. They exist to make the product feel
like an instrument with depth rather than a document — but the brief they are
held to is narrow, because "futuristic" is exactly the direction in which this
design system is easiest to wreck.

> **Before you edit any scene, read "the one thing worth knowing" in
> `HANDOFF.md`.** Uniforms must be reached through a ref on the material
> (`materialRef.current.uniforms`), never through the object the component built
> with `useMemo` and passed as a prop. The latter is a silent no-op — the writes
> land somewhere the GPU never reads, nothing errors, and the scene renders
> frozen at its initial uniform values. All five shader scenes shipped with this
> bug and were static until August 2026.

**Every canvas depicts something the model actually does.** Not one of them is
an abstract "AI" motif:

| Scene | Where | What it is |
| --- | --- | --- |
| `spectral-field` | `/` hero, behind the sample panel (ambient only, since 6 October 2026) | A spectrogram of speech as a ridgeline — the input, pictured the usual way. (Until October 2026 the served CNN/AASIST read log-Mel; SSL-AASIST reads the raw waveform through XLS-R, so captions say "the input", never "what the model reads".) |
| `intake-field` | `/upload` | The spectrogram as a *surface* — frequency across, time back, energy up — and the drop target itself. Idles on travelling waves, swells under the pointer, lifts while a file is dragged over it, and becomes your clip's envelope once decoded. |
| `waveform-display` | `/result` | The envelope of **your** clip, decoded in the browser and carried across from `/upload`. |
| `uncertainty-field` | `/result` | A lattice that holds its grid for a decisive reading and scatters as the reading nears the threshold. |
| `ambient-depth` | every route | One fixed backdrop giving the page depth behind the panels. |
| `signal-mark` | header | The wordmark's four bars, as geometry. |

**Interaction goes through a ref, never through state.** `intake-field` is the
only scene that responds to the pointer, and the coordinates reach it as
`pointerRef.current`, written by the DOM drop target and read by the render loop.
Routing a `pointermove` through `setState` would re-render the page sixty times a
second to deliver a value React never displays. The canvas stays
`pointer-events-none` — the DOM element above it does the listening, so the
geometry can never intercept a drag.

That constraint is what makes the layer defensible rather than decorative, and
it is also what lets the hero be brand indigo: a spectrogram pictures the
*input*, not a result, so rule 1 is intact.

**Eight rules, all enforced through `Stage`** — the single component every canvas
mounts through, so the guarantees are made once instead of remembered five
times:

1. **Colour comes from tokens.** `usePalette()` reads the resolved custom
   properties and hands shaders `THREE.Color`s. A hex literal in a shader is
   invisible to anyone auditing the palette and ignores the theme.
2. **The colour rules don't stop at the edge of a canvas.** Teal for chrome,
   the verdict scale for results. `uncertainty-field` is the only scene allowed
   a verdict colour, and it is the only one that renders inside a result.
3. **Never the only channel.** Every canvas is `aria-hidden`, takes no pointer
   events, and sits under DOM that carries the meaning by itself. No WebGL, no
   loss — `Stage` renders the `fallback` instead (the header falls back to the
   flat SVG mark). A page whose copy refers to a scene needs a real fallback:
   `/upload`'s "drop a clip on the surface" and `/result`'s clip waveform draw
   DOM bars without WebGL (added 7 October 2026, when the owner saw an empty
   page with hardware acceleration off).
4. **Stops when unwatched.** Offscreen or backgrounded canvases drop to
   `frameloop="never"`. Scenes advance their own clock from accumulated delta,
   so a paused field resumes where it left off instead of snapping.
5. **Reduced motion keeps the frame.** `frameloop="demand"` renders once and
   holds. Removing the scene entirely would punish the preference; freezing it
   honours it.
6. **Loaded after the page.** three.js is ~250 kB — more than the rest of the
   app combined. Everything is imported through `components/three/lazy.js`
   (`ssr: false`), which keeps First Load JS at ~108 kB instead of ~350 kB. An
   ambient layer must never decide how fast the interface appears.
7. **The visitor can stop it** (WCAG 2.2.2). The footer's "Pause background
   animation" control (`components/motion-switch.js`, remembered in
   `localStorage['soundsentinal-motion']`) holds every canvas still, exactly
   as reduced motion does. It was an icon button in the header until 7 October
   2026; the owner had it moved out.
8. **A slow machine sheds the layer, not the interface.** A software
   rasteriser (SwiftShader, llvmpipe, Basic Render — i.e. hardware
   acceleration off) counts as no WebGL. On real hardware a governor watches
   the frame rate: two seconds under 40 fps drops every canvas to 1× pixels,
   two more freezes them. The tier is page-wide and only steps down. The
   full-viewport backdrop renders without MSAA. The headless harness needs
   `localStorage['soundsentinal-3d'] = 'force'` to see any 3D at all.

**Placement is a legibility decision, not a composition one.** Two of these
moved during the build for exactly that reason: the hero field is masked and
weighted right so it never runs under the headline, and the uncertainty field
surrounds the meter panel rather than sitting behind it — the panel is dense
with small type at every height, so a point field inside it would sit under a
label wherever it was put. The opaque panel masks the middle, and the field is
only ever seen in the margins.

---

## Accessibility floor

Not negotiable, and already verified for every token pair in the system:

- Body and small text ≥ 4.5:1. UI borders and large text ≥ 3:1.
  `--text-faint` measures ~3.4:1 and is valid for decoration only, never text.
- The verdict scale is legible under protanopia, deuteranopia and tritanopia,
  and never the only channel carrying meaning.
- Visible keyboard focus everywhere, via the global `:focus-visible` ring.
- `prefers-reduced-motion` removes travel and keeps opacity, so nothing
  disappears for anyone who turns animation off.

---

## Open items

- **Refined 1 Oct 2026** (`UI-PLAN.md`): fonts actually load; fluid display
  type; the space scale and `band`; 44px hit areas; `Button href`; shared
  components; nav and footer; the home page's live "Measured, not claimed"
  error rates via `/api/model`; stale copy corrected (the served model was not
  trained with DP, and learned from LA + SpeechFake, not one corpus).
- **Migrated 17 Aug 2026.** `header.js` and the `/`, `/upload`, `/result` pages
  now use the tokens; the pre-system palette (`bg-white dark:bg-black`,
  `text-gray-600`) is gone from `src/` and `components/`. `/result` renders the
  calibration meter instead of a hand-rolled progress bar.
- `/result` still holds four demo buttons, now labelled *"Demo only — no model
  is connected yet"*. They come out when the page is wired to
  `spoof_probability` from the API.
- **The threshold on `/result` is a hardcoded 0.5** (`PLACEHOLDER_THRESHOLD`).
  The trainer computes a calibrated one from dev-set EER but `save_ckpt`
  discards it, so there is nothing to serve. Until that is persisted, the meter
  draws a marking the model doesn't actually use — the one place the design
  thesis is currently writing a cheque the backend can't cash.
- `/upload` enforces the limits it advertises (5 MB for wav/mp3/flac; 200 MB for
  mp4/m4a/mov/webm, whose soundtrack is extracted to WAV in the browser).
  Flask still needs `MAX_CONTENT_LENGTH` set to match.
- **`/upload` now decodes the clip in the browser** and shows duration, sample
  rate and channel count next to the waveform. The sample rate is parsed from
  the **container header**, not from the decoded buffer: `decodeAudioData`
  resamples to the AudioContext's rate, so `buffer.sampleRate` reports the
  output device (44.1 kHz) and the 16 kHz ASVspoof samples came back claiming
  44.1 kHz. When the header can't be read the readout is omitted rather than
  guessed. Decoding is a preview and never gates submission — FLAC support
  outside Chrome is patchy, and the server decodes separately with libsndfile.
- `upload → result` navigates via `sessionStorage` and does **not** call the
  model. The submit handler is where the `FormData` POST goes.
- The dark-mode toggle was previously inert: `theme.js` set a `.dark` class
  while `globals.css` keyed off `prefers-color-scheme`. Fixed via
  `@custom-variant dark` plus an inline pre-paint script in `layout.js`. The
  header's sun/moon icons switch via the `dark:` variant rather than from
  `darkMode` state, to avoid a hydration mismatch.
