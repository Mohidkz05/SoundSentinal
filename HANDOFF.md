# Handoff — frontend redesign session, 17–18 August 2026

Picking up from here: read this file, then `DESIGN.md` for the rules and
`CLAUDE.md` for the project. Nothing here is committed yet — it is all in the
working tree on `main`.

`npm run dev` → http://localhost:3000. `npx next build` passes clean as of the
end of this session (First Load JS ~109–112 kB, so three.js is still correctly
out of the initial bundle).

---

## The one thing worth knowing before you touch the 3D layer

**Shader uniforms must be reached through a ref on the material, never through
the object the component built.**

```jsx
// WRONG — silently does nothing
const uniforms = useMemo(() => ({ uTime: { value: 0 } }), []);
useFrame((_, d) => { uniforms.uTime.value += d; });          // goes nowhere
return <shaderMaterial uniforms={uniforms} ... />;

// RIGHT
const materialRef = useRef(null);
const initialUniforms = useMemo(() => ({ uTime: { value: 0 } }), []);
useFrame((_, d) => {
  const u = materialRef.current?.uniforms;
  if (!u) return;
  u.uTime.value += d;
});
return <shaderMaterial ref={materialRef} uniforms={initialUniforms} ... />;
```

The object the component holds is not the object the rendered material ends up
using, so every per-frame write lands somewhere the GPU never reads. **Nothing
errors.** The scene renders with whatever the uniforms were at creation time,
which looks exactly like a tuning problem rather than a wiring one.

**All five scenes had this bug, so the entire 3D layer had been static since it
was written.** Every "ambient drift", the spectrogram animation, the waveform
reveal, the uncertainty scatter — none of it was moving. What *did* work was
anything driven through an object transform (`groupRef.current.rotation.y = …`),
which is why the layer looked alive enough that nobody noticed.

All five are now fixed: `ambient-depth.js`, `spectral-field.js`,
`uncertainty-field.js`, `waveform-display.js`, `intake-field.js`.
`signal-mark.js` never had it (it uses `MeshStandardMaterial` and animates by
transform only).

**How it was found**, because the same method will work next time: set a
uniform that cannot be missed to a value that cannot be missed, every frame —
`u.uOpacity.value = 0.0` — and screenshot. If the thing is still visible, the
write is not reaching the GPU. Reasoning about the maths first cost hours;
this test took one minute and was conclusive.

### Verifying WebGL at all

The `chrome-devtools` MCP browser **has no WebGL**, so every canvas is absent in
its screenshots and the pages look like they have holes in them. That is the
harness, not the layout. To actually see the 3D:

```bash
google-chrome --headless=new --no-sandbox --disable-gpu-sandbox \
  --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader \
  --hide-scrollbars --window-size=1728,1000 --remote-debugging-port=9222 \
  --user-data-dir=/tmp/cdp-profile about:blank &
```

Then drive it over CDP. The scratchpad scripts from this session
(`shot2.mjs`, `drive.mjs`, `silhouette.py`) are in
`/tmp/claude-1000/-home-mohid-uni-SoundSentinal/<session>/scratchpad/` and will
be cleaned up — they are easy to rewrite, but two lessons from them are worth
keeping:

- **Disable the network cache** (`Network.setCacheDisabled`). Without it
  `Page.navigate` reuses cached HTML and therefore the previous build's chunk
  URLs, so every screenshot silently lags the source by one or more edits. This
  wasted real time.
- **Never A/B a local shader effect across two page loads.** The animation clock
  differs between them and the phase difference swamps whatever you are looking
  for. Either do both states in one session, or measure something global.

---

## What changed

### 1. Layout is full-bleed

`--shell` (the old 84rem max-width) is gone. There is now `--gutter`
(`clamp(1.25rem, 3vw, 3.5rem)`) and a `shell` utility that applies it with **no
max-width**. Also `edge` (gutter only) and `rule-full` (a hairline that spans
the bleed — at this width a rule is the main structural device, doing the job a
container edge used to).

The discipline that keeps it readable: **nothing inside runs the full width by
default.** Prose keeps a `ch` measure. What earns the width is the scale, the
waveform and the spectrogram, where width is resolution.

Where a heading needs a paragraph next to it, they go **side by side across the
row** (`lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]`), not stacked — a heading
with a measure-limited paragraph under it leaves most of a 1700px row empty.

Gotcha found the hard way: `max-w-[Nch]` on a display-sized heading resolves
against the *heading's* font size, not the body's, so it comes out far narrower
than it looks. Size hero text columns in rem.

### 2. The verdict scale no longer renders a gradient

Two passes on this. First the smooth gradient bar became an engraved graduated
scale; then the colour came off the graduations entirely.

- **`verdict-ramp`** is the continuous diverging violet↔orange, and **only
  `/design` renders it raw**, labelled as palette documentation.
- **Readings** render `components/ui/verdict-scale.js`: neutral graduations
  (`bg-muted` majors, `bg-line-strong` fines) through `graduation-fine` /
  `graduation-major` masks. Fifty fine ticks at two percentage points each, a
  full-height major every ten.
- **Colour lives only on the reading** — the needle, the big readout, the tier
  name.

Why: a continuous fill reads as a progress bar, i.e. a quantity accumulating
towards completion, which a probability is not; and a coloured *track* states a
verdict at every point on the axis including the ones the model said nothing
about. On a real instrument the face is neutral and the pointer carries the
state. Both colour rules from `DESIGN.md` survive intact, because they were
always rules about the reading rather than the track — hue still says which side
of the threshold, and the middle tiers are the desaturated tokens, so a needle
near the line still drains towards grey.

Mask geometry: both put the tick at 0.6% of the band's width and **centre it in
its cell**, which lands fine ticks on the odd percentages and majors on
5, 15 … 95. Every fifth fine tick is a major, the band is inset equally at both
ends instead of running out of ticks on the right, and the threshold annotation
falls in a gap rather than on top of a graduation.

The scale is shared by the meter, the home illustration and `/design`. Don't
hand-roll a second copy.

### 3. `/upload` is built around a new 3D scene

`components/three/intake-field.js` — a spectrogram *surface* (frequency across,
time back, energy up) built from 72 × 16 bars in perspective. It replaced the
flat row of bars because a row can only show the envelope; a surface can show
the envelope and its history, which is what the model actually consumes.

Four states, continuously blended:

| State | What happens |
| --- | --- |
| Idle | Two detuned travelling waves |
| Hover | A gaussian swell follows the pointer; the surface leans towards it |
| Drag | Energy lifts globally — the instrument reacting, not a dashed border recolouring |
| Decoded | The near row becomes the clip's real envelope, rows behind it hold attenuated copies |

**The whole panel is the drop target**, full bleed, ~78vh. Copy sits at the top
with a downward wash; the surface is the ground below it. (The first arrangement
put the text over the field and needed a wash heavy enough to bury the thing the
page is built around — in dark mode it erased it entirely.)

**The pointer goes through a ref, not state.** `pointerRef = useRef({ x, y,
active })` in `upload/page.js`, written by the DOM drop target's
`onPointerMove`, read by the render loop. A prop would mean re-rendering the
whole page sixty times a second to deliver a value React never displays. The
canvas is `pointer-events-none`, so the DOM element does the listening and the
geometry never intercepts a drag. `onDragOver` also calls `trackPointer`,
because a drag fires `dragover` rather than `pointermove` and the swell would
otherwise freeze mid-drag.

Two orientation bugs fixed in the shader, both worth not reintroducing:
`aCell.y` runs far→near, so it is flipped once at the top (`v = 1.0 - aCell.y`)
and everything downstream measures depth *into* the scene; and the pointer
arrives in screen coordinates where y grows downwards while the near edge of the
surface is at the *bottom* of the panel, so y is flipped when it reaches the
uniform.

Also `p.z = (0.5 - v) * DEPTH` — the camera looks down +z, so the near row needs
the larger z. And bars fade towards their own foot (`vRise`), because every bar
starts at the same floor and a thousand translucent quads overlapping there
turned the bottom of the field into fog.

### 4. `/result` has real content and a clip to describe

It used to know one thing about the clip — its filename — which is why it had
nothing to show but the number. `src/lib/clip.js` now carries the whole measured
clip across the page boundary in `sessionStorage`: name, size, duration, sample
rate, channels, and the envelope (quantised to two decimals, ~2 kB of JSON).
`sessionStorage` because it survives the navigation and a refresh, is scoped to
the tab, and is cleared when the tab closes — which matches what `/upload`
promises about the clip not being kept.

Sections now: the reading (huge readout + full-bleed scale) → the clip this
describes (3D waveform + duration/rate/channels/size) → where this reading falls
(the four tier bands as a table, with the current one marked) → what this number
is / where the threshold comes from / the model card → what this reading does not
tell you (unseen attacks, recording conditions, four seconds, who spoke) → demo
controls.

The model card is hardcoded in `MODEL_CARD` in `result/page.js`. **Keep it in
step with `ai_model/model.py`** — a stale model card is worse than none, because
it looks like provenance.

### 5. Home page

Same full-bleed treatment, sections separated by `rule-full` instead of panels.
The scale illustration now runs the full width, which is the honest way to show
it: that is the size the result page actually gives it.

---

## Files touched

```
new   components/ui/verdict-scale.js        the graduated scale, shared
new   components/three/intake-field.js      the upload surface
new   src/lib/clip.js                       upload → result handoff
new   HANDOFF.md                            this file

edit  src/app/globals.css                   --gutter, shell/edge/rule-full,
                                            verdict-ramp + graduation masks
edit  src/app/page.js                       full bleed, new scale section
edit  src/app/upload/page.js                rebuilt around the intake field
edit  src/app/result/page.js                rebuilt, much more content
edit  src/app/design/page.js                ramp vs graduated scale documented
edit  components/header.js                  shell
edit  components/ui/calibration-meter.js    uses VerdictScale
edit  components/three/{ambient-depth,spectral-field,uncertainty-field,
                        waveform-display}.js   uniform-ref fix
edit  DESIGN.md, CLAUDE.md                  rules for all of the above
```

`components/ui/calibration-meter.js` is now only used by `/design` — `/result`
composes the readout and `VerdictScale` directly, because at full bleed it wants
a different arrangement. Worth deciding whether the meter stays as a component
or gets folded away.

---

## Follow-up session, 21 August 2026

Everything in "Open" below was worked through. What changed, and what is left:

**Done.**

- **The whole vertical slice is wired.** `/upload` posts to a new
  `src/app/api/predict/route.js`, which forwards to Flask; `/result` renders the
  returned `spoof_probability` against the returned `threshold`. The demo
  buttons and the `useState(0.12)` are gone, and `clip.js` now carries the
  reading alongside the clip. Verified in the swiftshader browser: real FLAC
  upload → 48.7% reading → scale, waveform and stats all drawn from it.
- **The trainer persists its calibrated threshold** (plus dev metrics and a
  confusion matrix) and `app.py` applies it. Also fixed while in there: a
  resumed run reset `best_eer` to infinity, so the first epoch after a resume
  always overwrote `best.pth` however much worse it was.
- **The result page cannot show a number it doesn't have.** Arriving at
  `/result` directly renders a waiting state — no reading, no model card. The
  model card is built from the API response rather than a constant in the page.
- **Two real bugs the wiring exposed.** Two tier descriptions asserted which
  side of the threshold they were on, which was only true while the threshold
  was 0.5 — the first calibrated checkpoint (0.413) made the page contradict
  itself on screen. And below `sm` the header hid the *navigation* rather than
  the wordmark, so `/result` and `/design` were unreachable on a phone.
- **Mobile pass.** 360/390/768/1728 all clean, zero horizontal overflow on
  every page, and the full upload→result flow works at 390px. The 50-graduation
  scale is dense at that width but the majors still carry it; it reads as a
  band with legible major ticks rather than hatching. Left alone.
- **All five scenes verified animating**, canvas-region diffs over 2.5s, each
  scrolled into view. The uniform-ref fix holds everywhere.

**Not done, and why.**

- **Motion tuning (item 2 below) was not touched.** Measuring it under
  swiftshader turned out not to work: the frame-diff metric has a floor of
  ~1.9 mean (two captures taken back to back differ by that much), which swamps
  the signal. The drift constants are objectively slow — the backdrop's
  periods are 28 s and 37 s — so nothing is obviously too busy, but *how it
  feels* is a judgement for real hardware, which is item 1 and still open.
- **Item 1 itself.** Everything is still verified only under software
  rendering.

## Open, in rough priority order

1. **Look at it in a real browser.** Everything 3D was verified only under
   software rendering (swiftshader) at 1728×1000. Check the intake field's
   weight and the hover swell on real hardware, both themes, and on a laptop
   screen. The swell amplitude (`0.52`) and falloff (`17.0`) in
   `intake-field.js` are the two numbers to tune.
2. **Now that the 3D layer actually animates**, re-look at all five scenes.
   Every motion parameter in them was tuned against a static render, so the
   speeds and amplitudes are guesses. `--duration-sweep`-scale slowness is the
   house style; several of them may now be too busy.
3. **Mobile pass.** The layouts stack correctly and were checked at 390px, but
   that was before the `/upload` and `/result` rewrites. The 50-graduation
   scale is dense below ~400px — a coarser variant below `sm` was considered and
   skipped as not worth the coupling; revisit if it reads as hatching.
4. **Commit this.** It is a large working tree. Suggested split: the uniform-ref
   fix on its own (it is a bug fix and stands alone), then the design system
   changes, then the page rewrites.
5. **Then the actual product work**, unchanged from `CLAUDE.md`: wire the
   frontend to Flask (`src/app/api/predict/route.js` proxy), return
   `spoof_probability` + `threshold`, turn off `debug=True`, and train a model.
   `src/lib/clip.js` is deliberately shaped to become the POST response.

## Not done, deliberately

- No `/design` audit against the new tokens beyond the scale section.
- `README.md`'s create-next-app boilerplate is still there.
- The `verdict-ramp`/`graduation-*` utilities are documented in `DESIGN.md` but
  `/design` does not yet show the graduation masks as separate specimens.
