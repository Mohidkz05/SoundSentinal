'use client';

import React, { useEffect, useState } from 'react';
import Header from '../../components/header';
import Footer from '../../components/footer';
import { Button } from '../../components/ui/button';
import { SpectralField } from '../../components/three/lazy';
import { SampleReading } from '../../components/ui/sample-reading';
import { SectionHead } from '../../components/ui/section-head';
import { ErrorRates } from '../../components/ui/error-rates';
import { Notice } from '../../components/ui/notice';

/* Three steps, in the visitor's words. The third is the one competitors skip:
   every reading is shown beside how often this model is wrong. */
const STEPS = [
  {
    title: 'Upload a clip',
    body: 'A voice note, call recording or video. It is scored in memory and never stored.',
  },
  {
    title: 'Read the score',
    body: 'A needle on a scale, against the line where the model flags a clip. Near the line, it is unsure, and the page says so.',
  },
  {
    title: 'Weigh it',
    body: 'Every reading comes with how often this model is wrong. A low score is not proof a recording is real.',
  },
];

/* Masked rather than clipped, so the field has no edge anywhere. The panel
   sits on top of its middle, so it only shows in the margins around it. */
const FIELD_MASK =
  'radial-gradient(closest-side, #000 0%, #000 55%, transparent 100%)';

/* The served model's measured error rates, fetched rather than written
   down: a number on a home page is a claim, and this one is only allowed to be
   the one the live checkpoint was measured at (see /api/model). */
function useMeasured() {
  const [state, setState] = useState({ status: 'loading', measured: [] });
  useEffect(() => {
    let live = true;
    /* A hosted server asleep takes a minute or two to answer. Past a few
       seconds, say so — "loading" that long reads as broken. */
    const waking = setTimeout(
      () => live && setState((s) => (s.status === 'loading' ? { ...s, status: 'waking' } : s)),
      WAKING_AFTER_MS
    );
    fetch('/api/model')
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) =>
        live &&
        setState({
          status: 'ready',
          measured: d.measured ?? [],
        })
      )
      .catch(() => live && setState({ status: 'offline', measured: [] }));
    return () => {
      live = false;
      clearTimeout(waking);
    };
  }, []);
  return state;
}

const WAKING_AFTER_MS = 4000;

export default function Home() {
  const rates = useMeasured();
  return (
    /* `clip` rather than `hidden`: it stops the full-bleed field below from
       producing a horizontal scrollbar without creating a scroll container,
       which would break the sticky header. */
    <div className="flex min-h-screen flex-col overflow-x-clip">
      <Header />

      <main id="main" className="flex-1">
        {/* Hero ----------------------------------------------------------
            The hook is the product itself: a real reading of a real clip,
            next to a question in the visitor's own words. It is DOM, so it
            never depends on WebGL; the spectral field behind the panel is
            ambient and only ever seen in the panel's margins. The copy column
            is fixed and the panel takes the rest: with the copy on a 1fr
            column, a wide window left a gap the width of a third panel
            between them (9 October). The scale is what width improves. */}
        <section className="shell relative isolate grid items-center gap-y-12 pb-[var(--space-section)] pt-[calc(var(--space-section)*0.6)] lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:gap-x-16 xl:grid-cols-[minmax(0,32rem)_minmax(0,1fr)]">
          <div className="flex flex-col items-start gap-6">
            <h1 className="text-display text-balance">
              Heard a voice.
              <br />
              Not sure it&apos;s human?
            </h1>
            <p className="max-w-[40ch] text-body text-secondary">
              Upload the clip for a second opinion: a score against a
              calibrated threshold, and how often the model gets it wrong.
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <Button href="/upload" variant="primary" size="lg">
                Check a clip
              </Button>
              <Button href="#how-it-works" variant="quiet" size="lg">
                How it works
              </Button>
            </div>
          </div>

          <div className="relative">
            <div
              className="pointer-events-none absolute -inset-x-[12%] -inset-y-[18%] -z-10"
              style={{ maskImage: FIELD_MASK, WebkitMaskImage: FIELD_MASK }}
            >
              <SpectralField className="h-full w-full" opacity={0.75} />
            </div>
            <SampleReading />
          </div>
        </section>

        {/* How it works --------------------------------------------------- */}
        <section className="band">
          <h2 id="how-it-works" className="text-h2 scroll-mt-24">How it works</h2>
          <ol className="mt-[var(--space-head)] grid gap-[var(--space-group)] sm:grid-cols-3 sm:gap-12">
            {STEPS.map((step) => (
              <li key={step.title} className="flex max-w-[38ch] flex-col gap-[var(--space-stack)] border-t border-line-strong pt-5">
                <h3 className="text-h3">{step.title}</h3>
                <p className="text-small text-secondary">{step.body}</p>
              </li>
            ))}
          </ol>
        </section>

        {/* Measured, not claimed ------------------------------------------
            The trust signal competitors give as one "accuracy" figure. Here it
            is the two mistakes separately, per test set, for the live model —
            including the row that does not flatter it. */}
        <section className="band">
          <SectionHead title="Measured, not claimed">
            <p>
              How often the live model is wrong, on recordings it never trained
              on. A real voice flagged is an accusation; a fake let through is
              a miss.
            </p>
          </SectionHead>

          <div className="mt-[var(--space-head)] min-h-[16rem]">
            {rates.status === 'ready' && rates.measured.length > 0 ? (
              <>
                <ErrorRates measured={rates.measured} />
                <Notice className="mt-[var(--space-group)]">
                  It misses more clean, studio-quality fakes than noisy ones,
                  and some recent voice generators still get past it.{' '}
                  <strong className="font-semibold text-primary">
                    A low reading on a clean clip is not evidence it is real.
                  </strong>
                </Notice>
              </>
            ) : rates.status === 'loading' || rates.status === 'waking' ? (
              <p className="tick-label" aria-live="polite">
                {rates.status === 'waking'
                  ? 'Waking the model server — it sleeps when idle, and starting takes up to a minute…'
                  : 'Reading the model’s measurements…'}
              </p>
            ) : (
              <Notice>
                The model server isn&apos;t running, so its measured error rates
                can&apos;t be shown. They are only ever shown for the model that
                is actually live.
              </Notice>
            )}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
