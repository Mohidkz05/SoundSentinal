'use client';

import React, { useEffect, useState } from 'react';
import Header from '../../../components/header';
import Footer from '../../../components/footer';
import { Button } from '../../../components/ui/button';
import { Stat } from '../../../components/ui/stat';
import { Notice } from '../../../components/ui/notice';
import { Stepper } from '../../../components/ui/stepper';
import { SectionHead } from '../../../components/ui/section-head';
import { ErrorRates } from '../../../components/ui/error-rates';
import { VerdictScale } from '../../../components/ui/verdict-scale';
import { WaveformDisplay } from '../../../components/three/lazy';
import { readClip, formatBytes } from '../../lib/clip';
import {
  formatDuration,
  formatSampleRate,
  needsResampling,
  MODEL_SAMPLE_RATE,
  MODEL_WINDOW_SECONDS,
} from '../../lib/peaks';
import {
  confidence,
  formatConfidence,
  formatPercent,
  formatScore,
  readingScale,
  tierFor,
  tiersFor,
} from '../../lib/verdict';

/* The model card, built from what the API reported about the checkpoint that
   produced this reading — never written down here. Every row is a fact about
   the thing that made the number above it, and a reader who wants to discount
   the reading should be able to do it from this table alone. That is exactly
   why it isn't a constant: a hand-maintained card describes whichever run was
   current when someone last edited this file, which looks like provenance
   while being fiction. Rows the checkpoint doesn't know are dropped. */
function modelCard(model) {
  if (!model) return [];
  return [
    ['Input', model.input],
    ['Representation', model.representation],
    ['Network', model.network],
    ['Training corpus', model.corpus],
    ['Trained epochs', model.epoch != null ? String(model.epoch) : null],
    ['Dev EER', model.dev_eer != null ? formatPercent(model.dev_eer) : null],
    ['Privacy', model.privacy],
    ['Input processing', model.input_processing],
    ['Threshold from', model.threshold_source],
  ].filter(([, value]) => value);
}

const LIMITS = [
  ['Unseen generators', 'Voice-cloning services it never heard can pass. A low reading on one is not evidence the clip is real.'],
  ['Recording conditions', 'Noise, phone lines and compression shift scores for reasons unrelated to whether speech was synthesised.'],
  [`Only ${MODEL_WINDOW_SECONDS} seconds`, `Only the first ${MODEL_WINDOW_SECONDS} seconds are read; anything after that is never looked at.`],
  ['Not who spoke', 'It estimates whether speech was synthesised, not whose voice it is.'],
];

function bandRange(band) {
  if (!Number.isFinite(band.from)) return `below ${formatScore(band.to)}`;
  if (!Number.isFinite(band.to)) return `${formatScore(band.from)} and above`;
  return `${formatScore(band.from)} to ${formatScore(band.to)}`;
}

/* A native disclosure: keyboard and screen-reader behaviour for free. */
function Detail({ title, children }) {
  return (
    <details className="group border-b border-line">
      <summary className="flex min-h-[var(--hit)] cursor-pointer list-none items-center justify-between gap-4 py-3 text-body font-semibold text-primary hover:text-accent [&::-webkit-details-marker]:hidden">
        {title}
        <svg viewBox="0 0 24 24" className="h-4 w-4 flex-none transition-transform duration-[var(--duration-base)] group-open:rotate-180" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M6 9.5l6 6 6-6" />
        </svg>
      </summary>
      <div className="flex flex-col gap-[var(--space-stack)] pb-5 text-small text-secondary">{children}</div>
    </details>
  );
}

export default function ResultPage() {
  const [clip, setClip] = useState(null);
  /* Separate from `clip` because "we haven't looked yet" and "there is nothing
     there" are different screens, and sessionStorage is only readable after
     mount. Without this the waiting state flashes on every load. */
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setClip(readClip());
    setLoaded(true);
  }, []);

  const reading = clip?.reading ?? null;
  /* Before a reading exists the tiers are still described, against a
     neutral threshold; they are not a result until there is a score. */
  const { score, threshold, bandLow, bandMeasured, bandSource } = readingScale(reading);
  const hasReading = score != null;
  const tiers = tiersFor(threshold, bandLow);
  const tier = hasReading ? tierFor(score, tiers) : null;
  const clears = hasReading && score >= threshold;
  const measured = reading?.measured ?? [];
  const truncated =
    clip?.duration != null && clip.duration > MODEL_WINDOW_SECONDS;
  const card = modelCard(reading?.model);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main id="main" className="flex-1">
        {/* The reading ---------------------------------------------------
            The scale spans the full bleed, which is the one place on the site
            where that is a functional choice rather than a stylistic one: on a
            graduated scale, width is resolution. Across a whole screen, two
            readings a tenth of a unit apart are visibly different positions. */}
        <section className="shell pb-[var(--space-section)] pt-[calc(var(--space-section)*0.6)]">
          <div className="animate-rise flex flex-col gap-[var(--space-stack)]">
            <Stepper current={1} />
            <p className="text-small text-muted">
              {clip?.name ? (
                <>
                  Analysed <span className="tabular text-secondary">{clip.name}</span>
                </>
              ) : (
                'Analysed clip'
              )}
            </p>
          </div>

          {hasReading ? (
            <>
              <div className="mt-8 flex flex-wrap items-end gap-x-12 gap-y-6">
                <div className="flex flex-col gap-2">
                  <p className="tick-label">
                    Model confidence it is {clears ? 'AI generated' : 'real'}
                  </p>
                  <output
                    data-readout
                    className="block text-score font-medium"
                    style={{ color: tier.token }}
                  >
                    {formatConfidence(confidence(score, threshold))}
                  </output>
                </div>

                <div className="flex flex-col gap-2 pb-2">
                  <h1
                    className="text-h2 text-balance"
                    style={{ color: tier.token }}
                  >
                    {tier.headline}
                  </h1>
                  <p className="tick-label">
                    Score <span className="tabular">{formatScore(score)}</span>,{' '}
                    {clears ? 'above' : 'below'} the decision threshold of{' '}
                    <span className="tabular">{formatScore(threshold)}</span>
                  </p>
                </div>
              </div>

              <div className="mt-14">
                <VerdictScale
                  score={score}
                  threshold={threshold}
                  bandLow={bandLow}
                  height="h-16"
                />
              </div>

              <p className="mt-10 max-w-[68ch] text-body text-secondary">
                {tier.detail}
              </p>
              {/* The percentage is the model's certainty, and the page must not
                  let it pass for accuracy — see confidence() in verdict.js. */}
              <p className="mt-4 max-w-[68ch] text-small text-muted">
                Confidence is how sure the model is, not how often it is right.
              </p>

              {/* The next thing to do sits with the reading, not under eight
                  screens of explanation. */}
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <Button href="/upload" variant="primary">
                  Analyse another clip
                </Button>
                <Button href="#how-often-wrong" variant="quiet">
                  How often it is wrong
                </Button>
              </div>

              {/* An uncalibrated cutoff is still a cutoff, and the page draws it
                  as confidently either way. Saying where it came from is the
                  difference between a threshold and a number someone typed. */}
              {reading.threshold_calibrated === false && (
                <Notice className="mt-6">
                  This threshold is the default score of 0, not a calibrated
                  one — the checkpoint being served carries no operating point.
                  Treat the side of the line this reading falls on as
                  provisional.
                </Notice>
              )}
            </>
          ) : (
            /* No reading. The page does not invent one: a screen whose entire
               job is to report a measurement has nothing honest to show without
               it, so it says so and offers the way to get one. */
            <div className="mt-8 flex max-w-[62ch] flex-col items-start gap-5">
              <h1 className="text-h1 text-balance">
                {loaded ? 'No reading on this page' : 'Loading the reading…'}
              </h1>
              {loaded && (
                <>
                  <p className="text-body text-secondary">
                    Upload a clip first, and its reading will appear here.
                  </p>
                  <Button href="/upload" variant="primary" size="lg">
                    Analyse a clip
                  </Button>
                </>
              )}
            </div>
          )}
        </section>

        {/* The clip ------------------------------------------------------- */}
        {clip && (
        <section className="band">
          <div className="grid gap-[var(--space-group)] lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-14">
            <div className="flex flex-col gap-[var(--space-group)]">
              <h2 className="text-h2">The clip this describes</h2>
              {clip?.peaks ? (
                <>
                  <div className="h-40 w-full sm:h-52">
                    <WaveformDisplay className="h-full w-full" peaks={clip.peaks} />
                  </div>
                  <p className="max-w-[62ch] text-small text-muted">
                    Drawn in your browser; this picture never left it.
                    {truncated &&
                      ` The model read only the first ${MODEL_WINDOW_SECONDS} seconds of it.`}
                  </p>
                </>
              ) : (
                /* A clip whose preview decode failed — FLAC outside Chrome,
                   usually. The reading is still real; only the picture of it is
                   missing, and saying which is which matters. */
                <p className="max-w-[62ch] text-small text-muted">
                  This browser couldn&apos;t draw the file. The reading above
                  is unaffected.
                </p>
              )}
            </div>

            {clip && (
              <div className="grid grid-cols-2 gap-x-6 gap-y-[var(--space-group)] self-start">
                <Stat
                  label="Duration"
                  value={formatDuration(clip.duration ?? NaN)}
                  note={truncated ? `First ${MODEL_WINDOW_SECONDS} s analysed` : null}
                />
                {clip.sampleRate != null && (
                  <Stat
                    label="Sample rate"
                    value={formatSampleRate(clip.sampleRate)}
                    note={
                      needsResampling(clip.sampleRate)
                        ? `Resampled to ${formatSampleRate(MODEL_SAMPLE_RATE)}`
                        : 'Matches the model'
                    }
                  />
                )}
                {clip.channels != null && (
                  <Stat
                    label="Channels"
                    value={String(clip.channels)}
                    note={clip.channels > 1 ? 'Mixed down to mono' : 'Mono'}
                  />
                )}
                <Stat
                  label="File size"
                  value={formatBytes(clip.size)}
                  note={clip.extracted ? 'Soundtrack extracted in your browser' : null}
                />
              </div>
            )}
          </div>
        </section>
        )}

        {/* How often it is wrong ------------------------------------------
            Measured for this checkpoint at this threshold, and delivered by
            the API from evaluate.py's own reports — the server drops any
            report measured at a different threshold, so these numbers cannot
            describe a model other than the one that produced the reading. */}
        <section className="band">
          <SectionHead id="how-often-wrong" title="How often it is wrong">
            <p>
              Measured at this threshold, on recordings the model never trained
              on. A real voice flagged is an accusation; a fake let through is
              a miss.
            </p>
          </SectionHead>

          {measured.length > 0 ? (
            <div className="mt-[var(--space-head)]">
              <ErrorRates measured={measured} />
              <Notice className="mt-[var(--space-group)]">
                It misses more clean, studio-quality fakes than noisy ones, and
                some recent voice generators still get past it. If a clip
                sounds studio-clean, a low reading is not evidence it is real.
              </Notice>
            </div>
          ) : (
            <p className="mt-[var(--space-head)] max-w-[68ch] text-small text-muted">
              {hasReading
                ? 'No error rates were measured for this model at this threshold, so none are shown.'
                : 'Error rates appear once a clip has been analysed.'}
            </p>
          )}
        </section>

        {/* Limits: the weaknesses stay where the reading is (PRODUCT.md). */}
        <section className="band">
          <div className="grid gap-x-14 gap-y-[var(--space-group)] lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
            <h2 className="text-h2 text-balance">What it can&apos;t tell you</h2>
            <ul className="grid gap-x-12 gap-y-[var(--space-group)] sm:grid-cols-2">
              {LIMITS.map(([title, body]) => (
                <li key={title} className="flex flex-col gap-[var(--space-tight)]">
                  <p className="text-small font-semibold text-primary">{title}</p>
                  <p className="text-small text-secondary">{body}</p>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Details, closed by default: everything a reader who wants to check
            the reading needs, without making everyone else read it. */}
        <section className="band">
          <h2 className="text-h2">Details</h2>
          <div className="mt-[var(--space-group)] flex max-w-[72ch] flex-col border-t border-line">
            <Detail title="How the scale works">
              <p>
                {hasReading ? (
                  <>
                    This clip scored{' '}
                    <span className="tabular text-primary">{formatScore(score)}</span>{' '}
                    against a threshold of{' '}
                    <span className="tabular text-primary">{formatScore(threshold)}</span>.{' '}
                  </>
                ) : null}
                Positive scores lean synthetic, negative lean real; every 2.3
                points is ten times the odds. Confidence is 50% on the
                threshold and rises the further past it a score falls.
              </p>
              <ul className="flex flex-col gap-1">
                {tiers.map((band) => (
                  <li key={band.id}>
                    <span className="font-semibold text-primary">{band.label}</span>{' '}
                    <span className="tabular">({bandRange(band)})</span>: {band.detail}
                  </li>
                ))}
              </ul>
              <p>
                The uncertain band starts where genuine speech stops being
                typical
                {bandMeasured && bandSource
                  ? ` (${bandSource.charAt(0).toLowerCase() + bandSource.slice(1)})`
                  : ''}
                . The flagged band mirrors its width above the line, a
                convention: there is no held-out set of fakes to measure it on.
                {hasReading && !bandMeasured &&
                  ' This checkpoint carries no calibration data, so the band is a fixed width.'}
              </p>
            </Detail>

            <Detail title="Where the threshold comes from">
              <p>
                {!hasReading || bandMeasured
                  ? 'It is the score only a small, fixed share of genuine recordings the model never trained on reach. It was set before the model was tested on real-world audio, and never adjusted to fit that test.'
                  : 'It is the equal error rate point on a held-out partition, computed after training rather than chosen by hand.'}
              </p>
            </Detail>

            <Detail title="The model">
              {card.length > 0 ? (
                <dl className="flex flex-col">
                  {card.map(([term, value]) => (
                    <div
                      key={term}
                      className="flex flex-wrap justify-between gap-x-6 gap-y-1 border-b border-line py-2.5 last:border-0"
                    >
                      <dt className="tick-label">{term}</dt>
                      <dd className="text-small text-secondary">{value}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p>The model card is read from the checkpoint, so it appears once a clip has been analysed.</p>
              )}
            </Detail>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
