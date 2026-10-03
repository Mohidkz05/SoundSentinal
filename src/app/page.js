'use client';

import React, { useEffect, useState } from 'react';
import Header from '../../components/header';
import Footer from '../../components/footer';
import { Button } from '../../components/ui/button';
import { SpectralField } from '../../components/three/lazy';
import { VerdictScale } from '../../components/ui/verdict-scale';
import { SectionHead } from '../../components/ui/section-head';
import { ErrorRates } from '../../components/ui/error-rates';
import { Notice } from '../../components/ui/notice';

/* The thesis, stated three ways. These mirror the three rules in DESIGN.md —
   if one of them stops being true of the product, it should come off this page
   rather than quietly become marketing. */
const PRINCIPLES = [
  {
    title: 'A score, not a verdict',
    body: 'The model returns a score. We show you that number, and the band where it is unsure, instead of stamping REAL or FAKE over a clip.',
  },
  {
    title: 'The cutoff is on screen',
    body: 'Every detector has a decision threshold. Ours is calibrated on held-out data and drawn on the scale, so you can see how close a reading sits to it.',
  },
  {
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

/* The served model's measured error rates, fetched rather than written
   down: a number on a home page is a claim, and this one is only allowed to be
   the one the live checkpoint was measured at (see /api/model). */
function useMeasured() {
  const [state, setState] = useState({ status: 'loading', measured: [] });
  useEffect(() => {
    let live = true;
    fetch('/api/model')
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => live && setState({ status: 'ready', measured: d.measured ?? [] }))
      .catch(() => live && setState({ status: 'offline', measured: [] }));
    return () => {
      live = false;
    };
  }, []);
  return state;
}

export default function Home() {
  const rates = useMeasured();
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
        <section className="shell relative isolate grid gap-y-12 pb-[var(--space-section)] pt-[calc(var(--space-section)*0.9)] lg:min-h-[36rem] lg:grid-cols-[minmax(0,44rem)_minmax(0,1fr)] lg:gap-x-12">
          <div className="relative flex flex-col items-start gap-6">
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
              <Button href="/upload" variant="primary" size="lg">
                Analyse a clip
              </Button>
              <Button href="#how-it-works" variant="quiet" size="lg">
                How a reading works
              </Button>
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
        <section className="band">
          <SectionHead
            id="how-it-works"
            title="How a reading is shown"
            aside={<p className="tick-label">Illustration — not a result</p>}
          >
            <p>
              Every clip gets a score. The scale below is the one a result is
              read off: the score on the axis, the threshold drawn as a line,
              the band where genuine speech sometimes reaches bracketed under it.
            </p>
          </SectionHead>

          <div className="mt-[var(--space-head)]">
            <VerdictScale threshold={8} bandLow={2.5} height="h-16" />
          </div>

          <div className="mt-[var(--space-head)] grid gap-[var(--space-group)] lg:grid-cols-3 lg:gap-14">
            <p className="text-small text-secondary">
              A neutral face and a coloured pointer, the way an instrument is
              built. The graduations say nothing on their own — fifty of them,
              one per half point of the model&apos;s score — because the scale
              is an axis, not a verdict at every point along it.
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

        {/* Measured, not claimed ------------------------------------------
            The trust signal competitors give as one "accuracy" figure. Here it
            is the two mistakes separately, per test set, for the live model —
            including the row that does not flatter it. */}
        <section className="band">
          <SectionHead title="Measured, not claimed">
            <p>
              How often the model being served right now is wrong, at the
              threshold it is served at, on recordings it never trained on.
              Two mistakes, counted separately: a real voice flagged is an
              accusation, a fake let through is a miss.
            </p>
          </SectionHead>

          <div className="mt-[var(--space-head)] min-h-[16rem]">
            {rates.status === 'ready' && rates.measured.length > 0 ? (
              <>
                <ErrorRates measured={rates.measured} />
                <Notice className="mt-[var(--space-group)]">
                  Strong on real-world audio; weaker on clean synthetic speech.
                  The threshold is set for noisy, compressed recordings, and
                  between one in eight and one in five studio-clean fakes pass
                  it — and some recent text-to-speech models it never trained
                  on still pass most of the time. If a clip sounds
                  studio-clean, a low reading is not evidence it is real.
                </Notice>
              </>
            ) : rates.status === 'loading' ? (
              <p className="tick-label" aria-live="polite">Reading the model&apos;s measurements…</p>
            ) : (
              <Notice>
                The model server isn&apos;t running, so its measured error rates
                can&apos;t be shown. They are only ever shown for the model that
                is actually live.
              </Notice>
            )}
          </div>
        </section>

        {/* Principles ----------------------------------------------------- */}
        <section className="band">
          <SectionHead title="Three rules the interface keeps" />
          <div className="mt-[var(--space-head)] grid gap-[var(--space-group)] sm:grid-cols-3 sm:gap-12">
            {PRINCIPLES.map((p) => (
              <article key={p.title} className="flex flex-col gap-[var(--space-stack)]">
                <h3 className="text-h3 text-balance">{p.title}</h3>
                <p className="text-small text-secondary">{p.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
