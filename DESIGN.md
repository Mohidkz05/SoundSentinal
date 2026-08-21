# SoundSentinal — design system

**Live reference: `/design`.** It renders from the same CSS the product does, so
it cannot drift. Read it before building a screen.

Everything is defined in `src/app/globals.css`. Nothing here is decorative — if
you can't find a token for what you need, add it there first rather than writing
a one-off value in a component.

---

## The thesis

**An instrument, not a verdict machine.**

The detector returns a probability, not a fact — and right now it's a two-layer
CNN with no trained weights against the current architecture, so its real error
rate is unknown and likely 10–20% EER when measured. A UI that stamps a red
**FAKE** on a clip would be claiming a confidence the model does not have.

Three consequences run through every decision below:

1. The brand colour never renders a result.
2. The result scale loses its colour where the model is least certain.
3. The decision threshold is drawn on screen rather than hidden behind a label.

If a proposed change breaks one of these, it's the wrong change.

---

## Colour

Three families. **A colour does one job**; the moment the brand teal appears in a
result, or a verdict orange appears on a button, the system stops meaning
anything.

### Sentinel Teal — brand, actions, focus

Hue 190.3°, sampled from **viridis at 0.50**: the perceptually-uniform colormap
that renders the log-Mel spectrograms this model actually consumes. The brand
colour is drawn from the tool's own subject matter rather than picked for mood.

`--color-teal-50` … `--color-teal-950`. In components use the semantic aliases:
`--accent`, `--accent-hover`, `--accent-quiet`, `--accent-contrast`.

Chosen partly by elimination. The competitive field is crowded and predictable:

| Product | Signature colour |
| --- | --- |
| Pindrop | orange-red `#ff4b20` on charcoal |
| Reality Defender | periwinkle `#5a79bf` + orange gradient |
| Hiya / Loccus | violet `#6629e3`, electric blue `#4d65ff` |
| Deepware | crimson `#cd2653` |
| AI or Not | acid green `#96e071` on near-black |
| ElevenLabs classifier | plain blue `#2b7fff` |
| Resemble Detect | near-monochrome warm greys |

Nobody holds teal, and category research is explicit that fintech/security blue
has become saturated to the point of meaninglessness while deep teal still reads
as trustworthy. Teal is both differentiated and defensible.

### Petrol — surfaces and text

The same 190.3° hue at near-zero chroma. Neutrals that sit *under* the brand
rather than fighting it, and a dark theme that reads as an instrument panel
instead of default-dark-mode black.

Use `--canvas`, `--panel`, `--raised`, `--overlay`, `--line`, `--line-strong`
and `--text-primary/secondary/muted/faint`.

### Verdict scale — results only

Diverging **violet ↔ orange**: the two ends of `plasma`, the other
perceptually-uniform spectrogram colormap, and the same axis as ColorBrewer's
PuOr.

- **Hue** says which side of the decision threshold the reading falls on.
- **Chroma** says how far from it. The scale drains towards grey at the
  boundary, so "the model is unsure" is expressed by the colour draining out —
  not by a fourth hue nobody can name.

This structure was chosen on measurements, not taste. Candidate scales were
simulated through protanopia, deuteranopia and tritanopia (Machado 2009) and
scored on OKLab ΔE:

| Scale | Separation across threshold (worst case) |
| --- | --- |
| **violet ↔ orange (chosen)** | **ΔE ≥ 0.096** |
| indigo ↔ ember | ΔE ≥ 0.090 |
| blue ↔ ember | ΔE ≥ 0.085 |
| teal ↔ amber (brand-aligned) | ΔE ≥ 0.071 |
| teal ↔ ember (brand-aligned) | **ΔE ≥ 0.048 — fails** |

The brand-aligned options are the two worst. Teal and ember collapse to nearly
the same grey under protanopia, which is why **the brand colour cannot double as
the verdict colour** — that's a measured constraint, not a stylistic preference.

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
actually is: fifty fine ticks, one per two percentage points, with a full-height
major every ten.

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

**Archivo** for everything, at two widths. Headlines set expanded
(`font-variation-settings: "wdth" 112`) against normal-width body — the contrast
between display and body comes from the width axis rather than from dragging in
a second typeface.

**IBM Plex Mono** for every number, with tabular figures, because every number
this app shows is a measurement: probabilities, thresholds, sample rates,
durations. Tabular stops a changing readout from reflowing.

Use the generated utilities — `text-display`, `text-h1`, `text-h2`, `text-h3`,
`text-body`, `text-small`, `text-readout`, `text-tick`. They carry the paired
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
  a longer scale puts more distance between two readings a percentage point
  apart.
- The **waveform**, because it is a signal on a time axis and width buys
  visible detail.
- The **intake surface** on `/upload`, which is both the illustration and the
  drop target and so is the page.

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

## Controls

Four button variants in `components/ui/button.js`: `primary`, `secondary`,
`quiet`, `danger`.

**One `primary` per screen.** If two things are primary, neither is. Default to
`quiet` for anything that isn't the point of the screen.

Press feedback is a 1px downward nudge, never a scale — text stays on the pixel
grid instead of softening mid-press.

---

## Motion

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

**Tiers and threshold are different things and must not be conflated.** The four
tiers are fixed quarters of the probability range describing the reading itself,
so they don't move when the model is retrained. The threshold is the model's
calibrated operating point and moves between training runs. See
`src/lib/verdict.js`.

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
| `spectral-field` | `/` hero | A log-Mel spectrogram as a ridgeline — the representation `preprocess_waveform()` reduces every clip to. |
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
it is also what lets the hero be brand teal: a spectrogram is the model's
*input*, not a result, so rule 1 is intact.

**Six rules, all enforced through `Stage`** — the single component every canvas
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
   flat SVG mark).
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
- `/upload` enforces the limits it advertises (5 MB, wav/mp3/flac) client-side.
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
