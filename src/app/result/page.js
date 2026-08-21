'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import Header from '../../../components/header';
import { Button } from '../../../components/ui/button';
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
import { TIERS, tierFor, formatProbability } from '../../lib/verdict';

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
    ['Dev EER', model.dev_eer != null ? formatProbability(model.dev_eer) : null],
    ['Privacy', model.privacy],
    ['Threshold from', model.threshold_source],
  ].filter(([, value]) => value);
}

function Stat({ label, value, note }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="tick-label">{label}</p>
      <p className="tabular text-h3 leading-none text-primary">{value}</p>
      {note && <p className="text-small text-muted">{note}</p>}
    </div>
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
  const probability = reading?.spoof_probability ?? null;
  const threshold = reading?.threshold ?? null;
  const hasReading = probability != null;

  const tier = hasReading ? tierFor(probability) : null;
  const clears = hasReading && threshold != null && probability >= threshold;
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
            readings a percentage point apart are visibly different positions. */}
        <section className="shell pb-16 pt-14">
          <div className="animate-rise flex flex-col gap-3">
            <p className="tick-label">Step 2 of 2 · Reading</p>
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
                  <p className="tick-label">Synthetic likelihood</p>
                  <output
                    data-readout
                    className="block text-[clamp(4rem,11vw,9rem)] font-medium leading-[0.85] tracking-[-0.04em]"
                    style={{ color: tier.token }}
                  >
                    {formatProbability(probability)}
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
                    {formatProbability(threshold)}
                  </p>
                </div>
              </div>

              <div className="mt-14">
                <VerdictScale
                  probability={probability}
                  threshold={threshold}
                  height="h-16"
                />
              </div>

              <p className="mt-10 max-w-[68ch] text-body text-secondary">
                {tier.detail}
              </p>

              {/* An uncalibrated cutoff is still a cutoff, and the page draws it
                  as confidently either way. Saying where it came from is the
                  difference between a threshold and a number someone typed. */}
              {reading.threshold_calibrated === false && (
                <p className="mt-4 max-w-[68ch] text-small text-muted">
                  This threshold is the default 0.5, not a calibrated one — the
                  checkpoint being served carries no operating point. Treat the
                  side of the line this reading falls on as provisional.
                </p>
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
                  <Link href="/upload">
                    <Button variant="primary" size="lg">
                      Analyse a clip
                    </Button>
                  </Link>
                </>
              )}
            </div>
          )}
        </section>

        {/* The clip ------------------------------------------------------- */}
        {clip && (
        <section className="rule-full shell py-12">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-14">
            <div className="flex flex-col gap-5">
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
              <div className="grid grid-cols-2 gap-x-6 gap-y-8 self-start">
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
        <section className="rule-full shell py-12">
          {/* Heading left, its own explanation right. At full bleed a heading
              with a paragraph under it leaves most of the row empty, because
              the paragraph is held to a readable measure and the heading is
              not — pairing them across the row is what uses the width without
              stretching a line of text to 200 characters. */}
          <div className="grid gap-8 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)] lg:gap-14">
            <h2 className="text-h2 text-balance">Where this reading falls</h2>
            <p className="max-w-[80ch] text-small text-secondary">
              The bands are fixed quarters of the probability range and describe
              the reading itself, so they don&apos;t move when the model is
              retrained. The threshold does move — it is recomputed from
              held-out data on every run — which is why it is drawn separately
              rather than being one of these boundaries.
            </p>
          </div>

          <div className="mt-10 overflow-x-auto">
            <table className="w-full min-w-[44rem] border-collapse text-left">
              <thead>
                <tr className="border-b border-line">
                  <th scope="col" className="tick-label pb-3 pr-6 font-normal">
                    Band
                  </th>
                  <th scope="col" className="tick-label pb-3 pr-6 font-normal">
                    Range
                  </th>
                  <th scope="col" className="tick-label pb-3 font-normal">
                    What it means
                  </th>
                </tr>
              </thead>
              <tbody>
                {TIERS.map((band, i) => {
                  const low = i === 0 ? 0 : TIERS[i - 1].max;
                  const high = Math.min(band.max, 1);
                  const here = hasReading && band.id === tier.id;
                  return (
                    <tr
                      key={band.id}
                      className="border-b border-line align-top last:border-0"
                    >
                      <th
                        scope="row"
                        className="py-4 pr-6 font-semibold"
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
                      </th>
                      <td className="tabular py-4 pr-6 text-small text-secondary">
                        {formatProbability(low)} – {formatProbability(high)}
                      </td>
                      <td className="py-4 text-small text-secondary">
                        {band.detail}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* How to read it, and the model card ----------------------------- */}
        <section className="rule-full shell grid gap-10 py-12 lg:grid-cols-3 lg:gap-14">
          <div className="flex flex-col gap-3">
            <h2 className="text-h3">What this number is</h2>
            <p className="text-small text-secondary">
              A likelihood, not a fact.{' '}
              {hasReading ? (
                <>
                  The model scored this clip at{' '}
                  <span className="tabular text-primary">
                    {formatProbability(probability)}
                  </span>{' '}
                  on its synthetic-speech scale, against a decision threshold of{' '}
                  <span className="tabular text-primary">
                    {formatProbability(threshold)}
                  </span>
                  .
                </>
              ) : (
                <>
                  The model scores a clip from 0 to 1 on its synthetic-speech
                  scale, and that score is compared against a decision
                  threshold to produce a label.
                </>
              )}
            </p>
            <p className="text-small text-secondary">
              A reading near the threshold means the model was close to its own
              line — not that the clip is half fake. That is why the scale is
              graduated rather than filled: you read one position off it.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <h2 className="text-h3">Where the threshold comes from</h2>
            <p className="text-small text-secondary">
              It is the equal error rate point: the score at which the model
              wrongly flags a real clip exactly as often as it misses a fake
              one. It is computed on a held-out partition after training, not
              chosen by hand, and it is rarely 0.5 on imbalanced data.
            </p>
            <p className="text-small text-muted">
              It is stored with the weights it was computed for, so retraining
              moves the threshold and the reading together. A checkpoint that
              carries none is served at 0.5 and says so above.
            </p>
          </div>

          <div className="flex flex-col gap-3">
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
        <section className="rule-full shell py-12">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)] lg:gap-14">
            <h2 className="text-h2 text-balance">
              What this reading does not tell you
            </h2>
            <div className="grid gap-8 sm:grid-cols-2">
              <div className="flex flex-col gap-2">
                <p className="tick-label">Unseen attacks</p>
                <p className="text-small text-secondary">
                  The training corpus predates current neural codec and
                  commercial voice-cloning systems. A clip from one of those is
                  outside everything the model has seen, and a low reading is
                  not evidence it is real.
                </p>
              </div>
              <div className="flex flex-col gap-2">
                <p className="tick-label">Recording conditions</p>
                <p className="text-small text-secondary">
                  Noise, phone codecs and heavy compression all shift a clip
                  away from the clean corpus the model learned on. They push
                  readings around for reasons that have nothing to do with
                  whether the speech was synthesised.
                </p>
              </div>
              <div className="flex flex-col gap-2">
                <p className="tick-label">Four seconds</p>
                <p className="text-small text-secondary">
                  Only the first {MODEL_WINDOW_SECONDS} seconds are read. A clip
                  that is real for that window and synthetic afterwards reads as
                  real, because the rest was never looked at.
                </p>
              </div>
              <div className="flex flex-col gap-2">
                <p className="tick-label">Who spoke</p>
                <p className="text-small text-secondary">
                  This is not speaker verification. It estimates whether speech
                  was synthesised, and says nothing at all about whose voice it
                  is or whether the words were said.
                </p>
              </div>
            </div>
          </div>

          <p className="mt-12 max-w-[68ch] text-small text-muted">
            SoundSentinal is a university research project and the detector is
            still being trained and evaluated. Treat a reading as a signal worth
            following up, never as proof.
          </p>

          <div className="mt-8">
            <Link href="/upload">
              <Button variant="primary" size="lg">
                Analyse another clip
              </Button>
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
