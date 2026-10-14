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
import { Reveal, Rise, Stagger, StaggerItem } from '../../components/motion/reveal';
import { MEASURED } from '../lib/measured';

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

const isRow = (r) =>
  typeof r?.set === 'string' && Number.isFinite(r.real_flagged) && Number.isFinite(r.fakes_passed);

/* The served model's measured error rates. They render at once from the copy
   bundled in src/lib/measured.js — waiting on the model server left this panel
   empty for up to a minute after it had scaled to zero, which is what most
   visitors saw. The live answer (see /api/model) still replaces the copy when
   it arrives, so the table can't drift from the model actually served. */
function useMeasured() {
  const [measured, setMeasured] = useState(MEASURED);
  useEffect(() => {
    let live = true;
    fetch('/api/model')
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      /* Only well-formed rows replace the bundled copy: a row without its
         rates would otherwise render as a confident 0.0%. */
      .then((d) => {
        const rows = d?.measured;
        if (live && Array.isArray(rows) && rows.length && rows.every(isRow)) setMeasured(rows);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  return measured;
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor"
         strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 10h11M11 5.5 15.5 10 11 14.5" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg viewBox="0 0 20 20" className="h-4 w-4 flex-none" fill="none" stroke="currentColor"
         strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="4.5" y="9" width="11" height="7.5" rx="1.6" />
      <path d="M7 9V6.8a3 3 0 0 1 6 0V9" />
    </svg>
  );
}

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
        <section className="shell relative isolate grid items-center gap-y-12 pb-[calc(var(--space-section)*0.7)] pt-[calc(var(--space-section)*0.6)] lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:gap-x-16 xl:grid-cols-[minmax(0,32rem)_minmax(0,1fr)]">
          <Stagger above className="flex flex-col items-start gap-6">
            <StaggerItem as="h1" above className="text-display text-balance">
              Heard a voice.
              <br />
              Not sure it&apos;s <span className="text-accent">human?</span>
            </StaggerItem>
            <StaggerItem as="p" above className="max-w-[40ch] text-lead text-secondary">
              Upload the clip for a second opinion: a score against a
              calibrated threshold, and how often the model gets it wrong.
            </StaggerItem>
            <StaggerItem above className="mt-2 flex flex-wrap items-center gap-3">
              <Button href="/upload" variant="primary" size="lg">
                Check a clip
                <ArrowIcon />
              </Button>
              <Button href="#how-it-works" variant="quiet" size="lg">
                How it works
              </Button>
            </StaggerItem>
            {/* The one promise a nervous visitor wants before uploading
                anything, said where they decide to. Same claim as the footer. */}
            <StaggerItem as="p" above className="flex items-center gap-2 text-small text-muted">
              <LockIcon />
              Scored in memory and never stored.
            </StaggerItem>
          </Stagger>

          <Rise delay={0.15} className="relative">
            <div
              className="pointer-events-none absolute -inset-x-[12%] -inset-y-[18%] -z-10"
              style={{ maskImage: FIELD_MASK, WebkitMaskImage: FIELD_MASK }}
            >
              <SpectralField className="h-full w-full" opacity={0.75} />
            </div>
            <SampleReading />
          </Rise>
        </section>

        {/* How it works ---------------------------------------------------
            The page's colour band: the brand surface gives the long scroll a
            structure that hairlines alone did not. The numbers stay because
            the order is the point — you upload before you can read. */}
        <section className="surface-brand shell py-[var(--space-section)]">
          <h2 id="how-it-works" className="text-h2 scroll-mt-24">How it works</h2>
          <Stagger as="ol" className="mt-[var(--space-head)] grid gap-[var(--space-group)] sm:grid-cols-3 sm:gap-12">
            {STEPS.map((step, i) => (
              <StaggerItem as="li" key={step.title} className="flex max-w-[38ch] flex-col gap-[var(--space-stack)]">
                {/* Solid on the band: the step numbers are the band's accent. */}
                <span
                  className="grid h-10 w-10 place-items-center rounded-full bg-on-band font-mono text-small font-semibold text-band tabular shadow-[0_6px_16px_-6px_rgb(0_0_0/0.35)]"
                  aria-hidden="true"
                >
                  {i + 1}
                </span>
                <h3 className="mt-2 text-h3">{step.title}</h3>
                <p className="text-body text-secondary">{step.body}</p>
              </StaggerItem>
            ))}
          </Stagger>
        </section>

        {/* Measured, not claimed ------------------------------------------
            The trust signal competitors give as one "accuracy" figure. Here it
            is the two mistakes separately, per test set, for the live model —
            including the row that does not flatter it. */}
        <section className="band border-t-0">
          <SectionHead title="Measured, not claimed">
            <p>
              How often the live model is wrong, on recordings it never trained
              on. A real voice flagged is an accusation; a fake let through is
              a miss.
            </p>
          </SectionHead>

          <Reveal className="panel-raised mt-[var(--space-head)] p-5 sm:p-8">
            <ErrorRates measured={rates} />
            <Notice className="mt-[var(--space-group)]">
              It misses more clean, studio-quality fakes than noisy ones,
              and some recent voice generators still get past it.{' '}
              <strong className="font-semibold text-primary">
                A low reading on a clean clip is not evidence it is real.
              </strong>
            </Notice>
          </Reveal>
        </section>
      </main>

      <Footer />
    </div>
  );
}
