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

      <main className="flex-1">
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
                  <p className="tick-label">Model score</p>
                  <output
                    data-readout
                    className="block text-score font-medium"
                    style={{ color: tier.token }}
                  >
                    {formatScore(score)}
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
                    {clears ? 'Above' : 'Below'} the decision threshold of{' '}
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
                    You arrived here directly rather than through the upload
                    step, so there is no clip and no measurement to report.
                    Everything below still describes how a reading is produced
                    and what it is worth.
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
                    Decoded in your browser on the previous screen and carried
                    here in tab storage. It never went to a server.
                    {truncated &&
                      ` The model read only the first ${MODEL_WINDOW_SECONDS} seconds of it.`}
                  </p>
                </>
              ) : (
                /* A clip whose preview decode failed — FLAC outside Chrome,
                   usually. The reading is still real; only the picture of it is
                   missing, and saying which is which matters. */
                <p className="max-w-[62ch] text-small text-muted">
                  This browser couldn&apos;t decode the file to draw it. The
                  reading above is unaffected — the server decodes the clip
                  separately, with a different library.
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
                <Stat label="File size" value={formatBytes(clip.size)} />
              </div>
            )}
          </div>
        </section>
        )}

        {/* Where the reading falls ---------------------------------------- */}
        <section className="band">
          <SectionHead title="Where this reading falls">
              <p>
                The bands are placed around the threshold, so they move with
                it. The uncertain band starts where genuine speech stops being
                typical
                {bandMeasured && bandSource ? (
                  <>
                    {' '}
                    —{' '}
                    <span className="text-primary">
                      {bandSource.charAt(0).toLowerCase() + bandSource.slice(1)}
                    </span>
                  </>
                ) : null}
                . The flagged band mirrors its width above the line: a
                convention, because there is no held-out set of fakes to
                measure that edge on.
              </p>
              {hasReading && !bandMeasured && (
                <p className="text-muted">
                  The checkpoint being served carries no calibration data, so
                  this band is a fixed width around the threshold rather than a
                  measured one.
                </p>
              )}
          </SectionHead>

          <div className="mt-[var(--space-head)] overflow-x-auto">
            <table className="data-table sm:min-w-[44rem]">
              <thead>
                <tr>
                  <th scope="col">Band</th>
                  <th scope="col">Range</th>
                  <th scope="col" className="max-sm:hidden">
                    What it means
                  </th>
                </tr>
              </thead>
              <tbody>
                {tiers.map((band) => {
                  const here = hasReading && band.id === tier.id;
                  const range = !Number.isFinite(band.from)
                    ? `below ${formatScore(band.to)}`
                    : !Number.isFinite(band.to)
                      ? `${formatScore(band.from)} and above`
                      : `${formatScore(band.from)} to ${formatScore(band.to)}`;
                  return (
                    <tr key={band.id}>
                      <th
                        scope="row"
                        className="font-semibold"
                        style={{ color: here ? band.token : 'var(--text-muted)' }}
                      >
                        <span className="flex items-center gap-2.5">
                          <span
                            className="h-2.5 w-2.5 shrink-0 rounded-[var(--radius-tick)]"
                            style={{
                              background: here ? band.token : 'var(--line-strong)',
                            }}
                            aria-hidden="true"
                          />
                          {band.label}
                          {here && (
                            <span className="tick-label text-primary">
                              ← this reading
                            </span>
                          )}
                        </span>
                        {/* At phone width the meaning moves under the name
                            rather than into a column off the right edge. */}
                        <span className="mt-2 block text-small font-normal text-secondary sm:hidden">
                          {band.detail}
                        </span>
                      </th>
                      <td className="tabular whitespace-nowrap text-small text-secondary">
                        {range}
                      </td>
                      <td className="text-small text-secondary max-sm:hidden">
                        {band.detail}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* How often it is wrong ------------------------------------------
            Measured for this checkpoint at this threshold, and delivered by
            the API from evaluate.py's own reports — the server drops any
            report measured at a different threshold, so these numbers cannot
            describe a model other than the one that produced the reading. */}
        <section className="band">
          <SectionHead id="how-often-wrong" title="How often it is wrong">
            <p>
              Two mistakes, measured separately, because they cost different
              things: a genuine recording wrongly flagged is an accusation, a
              fake let through is a miss. Both were measured at the threshold
              this reading was compared against, on recordings the model never
              trained on and that played no part in choosing the threshold.
            </p>
          </SectionHead>

          {measured.length > 0 ? (
            <div className="mt-[var(--space-head)]">
              <ErrorRates measured={measured} />
            </div>
          ) : (
            <p className="mt-[var(--space-head)] max-w-[68ch] text-small text-muted">
              {hasReading
                ? 'No error rates have been measured for this model at this threshold, so none are shown. A rate measured at another threshold would describe a different operating point.'
                : 'The error rates are measured for the model and threshold that produced a reading, so they appear once a clip has been analysed.'}
            </p>
          )}

          {measured.length > 1 && (
            <p className="mt-[var(--space-group)] max-w-[72ch] text-small text-secondary">
              The rows disagree, and that is the finding. The threshold is set
              for real-world audio: noisy, compressed, recorded in rooms. On
              that, it rarely accuses a real speaker and rarely misses a fake.
              Clean synthetic speech is different. It scores far lower on this
              model, and a third or more of it passes even from systems the
              model trained on; from recent text-to-speech models it never
              heard, nearly all of it does. If a clip sounds studio-clean,{' '}
              <span className="text-primary">
                a low reading is not evidence that it is real
              </span>
              .
            </p>
          )}
        </section>

        {/* How to read it, and the model card ----------------------------- */}
        <section className="band grid gap-[var(--space-group)] lg:grid-cols-3 lg:gap-14">
          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">What this number is</h2>
            <p className="text-small text-secondary">
              A score, not a fact.{' '}
              {hasReading ? (
                <>
                  The model scored this clip{' '}
                  <span className="tabular text-primary">{formatScore(score)}</span>,
                  against a decision threshold of{' '}
                  <span className="tabular text-primary">
                    {formatScore(threshold)}
                  </span>
                  .
                </>
              ) : (
                <>
                  The model gives every clip a score, and that score is
                  compared against a decision threshold to produce a label.
                </>
              )}{' '}
              Positive scores lean synthetic, negative lean real, and 0 means
              the model found both equally likely. The scale is logarithmic:
              every 2.3 points is ten times the odds.
            </p>
            <p className="text-small text-secondary">
              A reading near the threshold means the model was close to its own
              line — not that the clip is half fake. That is why the scale is
              graduated rather than filled: you read one position off it.
            </p>
          </div>

          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">Where the threshold comes from</h2>
            {/* Two kinds of checkpoint exist: recalibrated by calibrate.py on
                held-out real speech (they carry the band's data), and the
                trainer's own dev-EER threshold. Describe the one being served. */}
            <p className="text-small text-secondary">
              {!hasReading || bandMeasured
                ? 'It is set on genuine speech the model never trained on: the score that only a small, fixed share of those real recordings reach. The model card names the recordings and the share. It is chosen before the model is tested on real-world audio, never adjusted to fit that test.'
                : 'It is the equal error rate point on a held-out partition: the score at which the model wrongly flags a real clip exactly as often as it misses a fake one. It is computed after training, not chosen by hand.'}
            </p>
            <p className="text-small text-muted">
              It is stored with the weights it was computed for, so retraining
              moves the threshold and the reading together. A checkpoint that
              carries none is served at 0 and says so above.
            </p>
          </div>

          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">The model</h2>
            {card.length > 0 ? (
              <dl className="mt-1 flex flex-col">
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
              <p className="text-small text-muted">
                The model card is read from the checkpoint that produced a
                reading, so it appears once a clip has been analysed. It is not
                written down on this page: a card kept by hand describes
                whichever training run someone last remembered to type in.
              </p>
            )}
          </div>
        </section>

        {/* Limits --------------------------------------------------------- */}
        <section className="band">
          <div className="grid gap-x-14 gap-y-[var(--space-group)] lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
            <h2 className="text-h2 text-balance">
              What this reading does not tell you
            </h2>
            <div className="grid gap-[var(--space-group)] sm:grid-cols-2">
              <div className="flex flex-col gap-[var(--space-tight)]">
                <p className="text-small font-semibold text-primary">Unseen attacks</p>
                <p className="text-small text-secondary">
                  The model learned from 2019-era and open-source generators.
                  Commercial voice-cloning services it never heard are outside
                  that, and a low reading on one of them is not evidence the
                  clip is real.
                </p>
              </div>
              <div className="flex flex-col gap-[var(--space-tight)]">
                <p className="text-small font-semibold text-primary">Recording conditions</p>
                <p className="text-small text-secondary">
                  Noise, phone codecs and heavy compression all shift a clip
                  away from the clean corpus the model learned on. They push
                  readings around for reasons that have nothing to do with
                  whether the speech was synthesised.
                </p>
              </div>
              <div className="flex flex-col gap-[var(--space-tight)]">
                <p className="text-small font-semibold text-primary">Four seconds</p>
                <p className="text-small text-secondary">
                  Only the first {MODEL_WINDOW_SECONDS} seconds are read. A clip
                  that is real for that window and synthetic afterwards reads as
                  real, because the rest was never looked at.
                </p>
              </div>
              <div className="flex flex-col gap-[var(--space-tight)]">
                <p className="text-small font-semibold text-primary">Who spoke</p>
                <p className="text-small text-secondary">
                  This is not speaker verification. It estimates whether speech
                  was synthesised, and says nothing at all about whose voice it
                  is or whether the words were said.
                </p>
              </div>
            </div>
          </div>

          {/* Secondary: the primary for this screen is the same action next
              to the reading. One primary per screen. */}
          <div className="mt-[var(--space-head)]">
            <Button href="/upload" variant="secondary" size="lg">
              Analyse another clip
            </Button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
