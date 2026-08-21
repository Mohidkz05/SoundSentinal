'use client';

import React from 'react';
import Link from 'next/link';
import Header from '../../components/header';
import { Button } from '../../components/ui/button';
import { SpectralField } from '../../components/three/lazy';
import { VerdictScale } from '../../components/ui/verdict-scale';

/* The thesis, stated three ways. These mirror the three rules in DESIGN.md —
   if one of them stops being true of the product, it should come off this page
   rather than quietly become marketing. */
const PRINCIPLES = [
  {
    label: 'Reading',
    title: 'A probability, not a verdict',
    body: 'The model returns a likelihood. We show you that number instead of stamping REAL or FAKE over a clip it was only ever 71% sure about.',
  },
  {
    label: 'Threshold',
    title: 'The cutoff is on screen',
    body: 'Every detector has a decision threshold. Ours is calibrated on held-out data and drawn on the scale, so you can see how close a reading sits to it.',
  },
  {
    label: 'Uncertainty',
    title: 'The colour drains when we are unsure',
    body: 'The scale loses its saturation near the boundary. Where the model is least confident, the interface stops looking confident too.',
  },
];

/* The field is masked rather than clipped so it has no edge anywhere — a
   rectangle of animation with a visible border reads as an embedded video. It
   is centred on its own box, which since the hero became a grid is already
   clear of the text column; it no longer has to be pushed off-centre to stay
   away from the headline. */
const FIELD_MASK =
  'radial-gradient(80% 92% at 50% 50%, #000 0%, #000 40%, transparent 78%)';

export default function Home() {
  return (
    /* `clip` rather than `hidden`: it stops the full-bleed field below from
       producing a horizontal scrollbar without creating a scroll container,
       which would break the sticky header. */
    <div className="flex min-h-screen flex-col overflow-x-clip">
      <Header />

      <main className="flex-1">
        {/* Hero ---------------------------------------------------------- */}
        {/* The hero is a two-column grid rather than text with a picture
            behind it: the text column is pinned to a readable measure and the
            field gets everything to its right, bleeding to the viewport edge so
            the page doesn't look like a document with an image pasted into it.
            Below `lg` the field drops into a band under the call to action —
            there is no room beside the text on a narrow screen, and running it
            behind body copy would cost legibility for decoration. */}
        {/* The text column is sized in rem, not ch: `ch` on a display-sized
            heading resolves against the display font size and came out far
            narrower than the headline needs, breaking "not a verdict machine."
            across three lines. 44rem is the width that line wants at 68px. */}
        <section className="shell relative isolate grid gap-y-12 pb-20 pt-16 sm:pt-24 lg:min-h-[36rem] lg:grid-cols-[minmax(0,44rem)_minmax(0,1fr)] lg:gap-x-12">
          <div className="relative flex flex-col items-start gap-6">
            <p className="tick-label">Audio authenticity analysis</p>

            <h1 className="text-display text-balance">
              An instrument,
              <br />
              not a verdict machine.
            </h1>

            <p className="max-w-[46ch] text-body text-secondary">
              Upload a voice clip and get a calibrated reading of how likely it
              is to be synthetic — measured against the model&apos;s own
              decision threshold, with its uncertainty shown rather than hidden.
            </p>

            <div className="mt-2 flex flex-wrap items-center gap-3">
              <Link href="/upload">
                <Button variant="primary" size="lg">
                  Analyse a clip
                </Button>
              </Link>
              <Link href="/design">
                <Button variant="quiet" size="lg">
                  Design system
                </Button>
              </Link>
            </div>

            {/* Naming the field is the honest move: it stops being decoration
                and becomes a caption for what the model consumes. */}
            <p className="tick-label mt-auto max-w-[34ch] pt-10">
              Log-Mel spectrogram — the representation every clip is reduced to
              before the model reads it
            </p>
          </div>

          <div className="relative min-h-[15rem] lg:min-h-0">
            <div
              className="pointer-events-none absolute inset-y-0 left-[calc(50%-50vw)] right-[calc(50%-50vw)] lg:left-0 lg:right-[calc(50%-50vw)]"
              style={{ maskImage: FIELD_MASK, WebkitMaskImage: FIELD_MASK }}
            >
              <SpectralField className="h-full w-full" opacity={0.9} />
            </div>
          </div>
        </section>

        {/* The scale, explained ------------------------------------------
            The scale runs the full bleed, which is the honest way to show it:
            on a graduated scale width is resolution, so this is what it looks
            like at the size the result page actually gives it. */}
        <section className="rule-full shell py-14">
          <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2">
            <h2 className="text-h2">How a reading is shown</h2>
            <p className="tick-label">Illustration — not a result</p>
          </div>

          <div className="mt-10">
            <VerdictScale threshold={0.5} height="h-16" />
          </div>

          <div className="mt-10 grid gap-8 lg:grid-cols-3 lg:gap-14">
            <p className="text-small text-secondary">
              A neutral face and a coloured pointer, the way an instrument is
              built. The graduations say nothing on their own — fifty of them,
              one per two percentage points — because the scale is an axis, not
              a verdict at every point along it.
            </p>
            <p className="text-small text-secondary">
              Colour belongs to the reading alone. The needle takes a diverging
              violet↔orange: hue says which side of the threshold the clip fell
              on, and it drains towards grey near the line, so a reading the
              model was unsure about does not look confident.
            </p>
            <p className="text-small text-secondary">
              The brand teal never renders a result. It measured as
              indistinguishable from the warm end under protanopia, which makes
              it unusable for this and is why the two palettes are separate.
            </p>
          </div>
        </section>

        {/* Principles ----------------------------------------------------- */}
        <section className="rule-full shell py-14">
          <h2 className="text-h2 text-balance">Three rules the interface keeps</h2>
          <div className="mt-10 grid gap-10 sm:grid-cols-3 sm:gap-12">
            {PRINCIPLES.map((p) => (
              <article key={p.label} className="flex flex-col gap-3">
                <p className="tick-label">{p.label}</p>
                <h3 className="text-h3 text-balance">{p.title}</h3>
                <p className="text-small text-secondary">{p.body}</p>
              </article>
            ))}
          </div>
        </section>

        {/* Honesty note --------------------------------------------------- */}
        <section className="rule-full shell py-12">
          <p className="max-w-[68ch] text-small text-muted">
            SoundSentinal is a university research project. The detector is
            still being trained and evaluated, so readings should be treated as
            experimental — not as evidence.
          </p>
        </section>
      </main>
    </div>
  );
}
