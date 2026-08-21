'use client';

import React from 'react';
import { Button } from '../../../components/ui/button';
import { CalibrationMeter } from '../../../components/ui/calibration-meter';
import { VerdictScale } from '../../../components/ui/verdict-scale';
import { useTheme } from '../../../components/theme';
import {
  SpectralField,
  UncertaintyField,
  WaveformDisplay,
} from '../../../components/three/lazy';

/**
 * The living design system reference.
 *
 * Everything the frontend is allowed to use appears here once. If a value isn't
 * on this page it isn't in the system — add it here first, then use it. Keeping
 * it as a route rather than a static document means it can't drift from the
 * tokens: it renders from the same CSS the product does.
 */

export default function DesignSystem() {
  const { darkMode, toggleDarkMode } = useTheme() ?? {};

  return (
    <div className="shell min-h-screen py-16">
      <div className="max-w-[76rem]">
        <header className="mb-20">
          <p className="tick-label mb-4">SoundSentinal — design system v1</p>
          <h1
            className="mb-5 text-h1 font-bold"
            style={{ fontVariationSettings: '"wdth" 112' }}
          >
            An instrument, not a verdict machine
          </h1>
          <p className="max-w-prose text-[var(--text-secondary)]">
            The detector reports a probability, not a fact. Every decision here
            follows from that: the brand colour never renders a result, the
            result scale loses its colour where the model is least sure, and the
            decision threshold is drawn on screen instead of hidden behind a
            label.
          </p>
          <div className="mt-8">
            <Button size="sm" variant="quiet" onClick={toggleDarkMode}>
              {darkMode ? 'Switch to light' : 'Switch to dark'}
            </Button>
          </div>
        </header>

        <Section
          n="Colour"
          title="Three families, three jobs"
          note="A colour does one job. Teal is the product's voice and never renders a result; the diverging scale renders results and never appears as chrome; danger means something broke."
        >
          <Swatches
            label="Sentinel Teal — brand, actions, focus"
            sub="Hue 190.3°, sampled from viridis at 0.50 — the colormap that renders the log-Mel spectrograms the model reads."
            items={[
              ['50', '#eefaf8'], ['200', '#b2e5e1'], ['300', '#84d3ce'],
              ['400', '#53bdb7'], ['500', '#1ca5a0'], ['600', '#078b86'],
              ['700', '#056e6a'], ['800', '#03524f'], ['900', '#013735'],
            ]}
          />
          <Swatches
            label="Petrol — surfaces and text"
            sub="The same hue at near-zero chroma. Greys that sit under the brand instead of fighting it."
            items={[
              ['50', '#f7fbfb'], ['100', '#eff5f4'], ['300', '#d7e0df'],
              ['500', '#889594'], ['600', '#5f6c6b'], ['700', '#42504f'],
              ['800', '#2a3938'], ['900', '#142120'], ['950', '#0a1514'],
            ]}
          />
          <div className="mt-10">
            <p className="mb-1 text-small font-semibold">
              Verdict scale — results only
            </p>
            <p className="mb-4 max-w-prose text-small text-[var(--text-muted)]">
              Diverging violet ↔ orange: the two ends of <code className="font-mono">plasma</code>,
              and the same axis as ColorBrewer&apos;s PuOr. Hue says which side of the
              threshold the reading falls on; chroma says how far from it, so the
              scale drains towards grey exactly where the model is least certain.
              Measured across normal, protanope, deuteranope and tritanope vision,
              separation across the threshold stays at ΔE ≥ 0.096.
            </p>
            {/* The raw ramp. This is the only place it is drawn continuous —
                here it documents the palette. A reading is never rendered as a
                gradient bar, because a continuous fill reads as a progress bar
                and a probability is not a quantity accumulating towards
                completion. */}
            <div className="verdict-ramp h-14 w-full rounded-[var(--radius-sm)]" />
            <p className="tick-label mt-2">
              The ramp — palette documentation, never a reading
            </p>

            <p className="mb-4 mt-10 max-w-prose text-small text-[var(--text-muted)]">
              Readings render that ramp through a graduation mask instead: fifty
              fine ticks, one per two percentage points, with a full-height major
              every ten. The mask changes where the colour is painted, never
              which colour it is, so both rules survive — hue still says which
              side of the threshold, chroma still drains at the boundary — and
              the scale becomes something you read a position off.
            </p>
            <VerdictScale threshold={0.5} probability={0.71} />
          </div>
        </Section>

        <Section
          n="Type"
          title="Archivo, two widths"
          note="One superfamily. Headlines set expanded at wdth 112 against normal-width body — the contrast between roles comes from width, not from a second typeface. Numbers always take IBM Plex Mono with tabular figures, because every number here is a measurement."
        >
          <div className="space-y-6">
            <TypeRow spec="display · 68px / 0.94 / -0.03em · wdth 112">
              <span className="text-display font-bold leading-[0.94]"
                    style={{ fontVariationSettings: '"wdth" 112' }}>
                Is it real?
              </span>
            </TypeRow>
            <TypeRow spec="h2 · 32px / 1.1 / -0.02em">
              <span className="text-h2 font-bold"
                    style={{ fontVariationSettings: '"wdth" 112' }}>
                Upload a clip to analyse
              </span>
            </TypeRow>
            <TypeRow spec="body · 16px / 1.6">
              <span className="text-[var(--text-secondary)]">
                The model compares your clip against the synthesis artefacts it
                learned during training.
              </span>
            </TypeRow>
            <TypeRow spec="readout · 56px · mono, tabular">
              <span data-readout className="text-readout font-medium">
                71.4%
              </span>
            </TypeRow>
            <TypeRow spec="tick · 11px · mono, 0.14em, uppercase">
              <span className="tick-label">Threshold · 16 kHz · 4.00 s</span>
            </TypeRow>
          </div>
        </Section>

        <Section
          n="Surface"
          title="Elevation, rounding, edges"
          note="On paper, elevation is a shadow. On the dark panel it is surface lightness plus a 1px top highlight — a big black shadow on a dark ground reads as mud. Both are the same token, defined twice."
        >
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="panel p-5">
              <p className="tick-label mb-2">panel</p>
              <p className="text-small text-[var(--text-secondary)]">
                Default container. radius-lg, 14px.
              </p>
            </div>
            <div className="panel-raised p-5">
              <p className="tick-label mb-2">panel-raised</p>
              <p className="text-small text-[var(--text-secondary)]">
                Sits above the page. Dropzone, result card.
              </p>
            </div>
            <div className="rounded-[var(--radius-xl)] border border-[var(--line-strong)] bg-[var(--overlay)] p-5 shadow-[var(--shadow-lifted)]">
              <p className="tick-label mb-2">lifted</p>
              <p className="text-small text-[var(--text-secondary)]">
                Modals only. radius-xl, 20px.
              </p>
            </div>
          </div>
          <div className="mt-6 flex flex-wrap items-end gap-3">
            {[['tick', 2], ['xs', 4], ['sm', 6], ['md', 10], ['lg', 14], ['xl', 20]].map(
              ([name, px]) => (
                <div key={name} className="text-center">
                  <div
                    className="mb-1.5 h-14 w-14 border border-[var(--line-strong)] bg-[var(--panel)]"
                    style={{ borderRadius: `${px}px` }}
                  />
                  <span className="tick-label">{name}</span>
                </div>
              )
            )}
          </div>
          <div className="rule-etched my-8" />
        </Section>

        <Section
          n="Controls"
          title="One primary per screen"
          note="If two things are primary, neither is. Press feedback is a 1px nudge, never a scale — text stays on the pixel grid instead of softening mid-press."
        >
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary">Analyse clip</Button>
            <Button variant="secondary">Choose a different file</Button>
            <Button variant="quiet">Back</Button>
            <Button variant="danger">Remove</Button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Button variant="primary" size="lg">Large</Button>
            <Button variant="primary" size="sm">Small</Button>
            <Button variant="primary" loading>Analysing</Button>
            <Button variant="primary" disabled>Disabled</Button>
          </div>
        </Section>

        <Section
          n="Signature"
          title="The calibration meter"
          note="The element the product is remembered by. No competitor publishes its decision threshold; drawing it turns an unexplained verdict into a reading you can argue with. Three channels carry the result — needle position, tier name, numeric readout — so colour is never load-bearing alone."
        >
          <div className="panel-raised px-7 pt-12 pb-8">
            <CalibrationMeter probability={0.714} threshold={0.5} />
          </div>
        </Section>

        <Section
          n="Motion"
          title="One gesture, spent once"
          note="Everything uses the instrument easing — decisive, no bounce. The needle easing overshoots 4% and settles, the way a real meter does, and is reserved for the reading. Using it elsewhere would spend the gesture and it would stop meaning 'a measurement landed'."
        >
          <table className="w-full text-left text-small">
            <thead>
              <tr className="border-b border-[var(--line)]">
                <th className="tick-label py-2">Token</th>
                <th className="tick-label py-2">Value</th>
                <th className="tick-label py-2">Use</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[var(--text-secondary)]">
              {[
                ['tap', '100ms', 'press states'],
                ['fast', '160ms', 'hover, focus'],
                ['base', '240ms', 'panels, disclosure'],
                ['slow', '420ms', 'page entrance'],
                ['sweep', '900ms', 'needle travel'],
                ['ease-instrument', '0.2, 0.8, 0.2, 1', 'everything'],
                ['ease-needle', '0.34, 1.28, 0.44, 1', 'the reading only'],
              ].map(([a, b, c]) => (
                <tr key={a} className="border-b border-[var(--line)]">
                  <td className="py-2">{a}</td>
                  <td className="py-2">{b}</td>
                  <td className="py-2 font-sans">{c}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section
          n="Dimension"
          title="The 3D layer"
          note="Every canvas depicts something the model actually does — the spectrogram it reads, the waveform you gave it, the uncertainty around its threshold. None of them is decoration with a subject bolted on, and none is load-bearing: the DOM underneath carries the meaning, and the geometry is an enhancement of it."
        >
          <div className="panel mb-8 overflow-hidden">
            <div className="h-44">
              <SpectralField className="h-full w-full" opacity={0.9} />
            </div>
            <p className="border-t border-[var(--line)] px-5 py-3 text-small text-[var(--text-muted)]">
              <span className="font-semibold text-[var(--text-primary)]">
                Spectral field
              </span>{' '}
              — a log-Mel spectrogram as a ridgeline. Brand teal, because it is
              the model&apos;s input format and not a result.
            </p>
          </div>

          <div className="panel mb-8 overflow-hidden">
            <div className="h-40">
              <WaveformDisplay className="h-full w-full" />
            </div>
            <p className="border-t border-[var(--line)] px-5 py-3 text-small text-[var(--text-muted)]">
              <span className="font-semibold text-[var(--text-primary)]">
                Waveform display
              </span>{' '}
              — idling here; on <code className="font-mono">/upload</code> it
              shows the envelope of the clip you chose, decoded in the browser.
            </p>
          </div>

          <div className="mb-10 grid gap-4 sm:grid-cols-2">
            {[
              { p: 0.03, caption: 'Decisive reading — the lattice holds' },
              { p: 0.49, caption: 'On the threshold — order comes apart' },
            ].map(({ p, caption }) => (
              <div key={p} className="panel overflow-hidden">
                <div className="h-40 bg-[var(--canvas)]">
                  <UncertaintyField className="h-full w-full" probability={p} threshold={0.5} />
                </div>
                <p className="border-t border-[var(--line)] px-5 py-3 text-small text-[var(--text-muted)]">
                  {caption}
                </p>
              </div>
            ))}
          </div>

          <table className="w-full text-left text-small">
            <thead>
              <tr className="border-b border-[var(--line)]">
                <th className="tick-label py-2">Rule</th>
                <th className="tick-label py-2">Why</th>
              </tr>
            </thead>
            <tbody className="text-[var(--text-secondary)]">
              {[
                ['Colour comes from tokens', 'Shaders read the resolved custom properties, so both themes and any future palette change reach the geometry.'],
                ['Teal for chrome, verdict scale for results', 'The colour rules do not stop at the edge of a canvas. Only the uncertainty field is allowed a verdict colour.'],
                ['Never the only channel', 'Every canvas is aria-hidden and takes no pointer events. Nothing is lost without WebGL.'],
                ['Stops when unwatched', 'Offscreen or backgrounded canvases drop to frameloop "never" rather than rendering to nobody.'],
                ['Reduced motion keeps the frame', 'One render on demand, then still — the composition survives, the movement does not.'],
                ['Loaded after the page', 'three.js is ~250 kB and arrives in a deferred chunk, so it never decides how fast the interface appears.'],
              ].map(([rule, why]) => (
                <tr key={rule} className="border-b border-[var(--line)] align-top">
                  <td className="w-[16rem] py-2.5 pr-4 font-semibold text-[var(--text-primary)]">
                    {rule}
                  </td>
                  <td className="py-2.5">{why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      </div>
    </div>
  );
}

function Section({ n, title, note, children }) {
  return (
    <section className="mb-20">
      <p className="tick-label mb-3">{n}</p>
      <h2
        className="mb-3 text-h2 font-bold"
        style={{ fontVariationSettings: '"wdth" 112' }}
      >
        {title}
      </h2>
      <p className="mb-8 max-w-prose text-small text-[var(--text-muted)]">
        {note}
      </p>
      {children}
    </section>
  );
}

function Swatches({ label, sub, items }) {
  return (
    <div className="mb-10">
      <p className="mb-1 text-small font-semibold">{label}</p>
      <p className="mb-4 max-w-prose text-small text-[var(--text-muted)]">
        {sub}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {items.map(([step, hex]) => (
          <div key={step} className="w-[4.5rem]">
            <div
              className="mb-1.5 h-14 rounded-[var(--radius-xs)] border border-[var(--line)]"
              style={{ background: hex }}
            />
            <p className="font-mono text-[0.625rem] text-[var(--text-muted)]">{step}</p>
            <p className="font-mono text-[0.625rem] text-[var(--text-faint)]">{hex}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TypeRow({ spec, children }) {
  return (
    <div className="border-b border-[var(--line)] pb-5">
      <p className="tick-label mb-2">{spec}</p>
      {children}
    </div>
  );
}
