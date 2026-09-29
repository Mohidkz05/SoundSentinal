'use client';

import React, { useEffect, useState } from 'react';
import {
  SCALE_NUMERALS,
  formatScore,
  position,
  tierFor,
  tiersFor,
} from '../../src/lib/verdict';

/**
 * The graduated scale a reading is read off.
 *
 * Two decisions here, both arrived at by removing something.
 *
 * **It is not a filled bar.** A continuous fill reads as a progress bar — a
 * quantity accumulating towards completion — and a score is not that. It also
 * drags the eye along its whole length when the only thing carrying the result
 * is one point on it. So the scale is engraved instead: fifty fine graduations
 * at half a score unit each, a full-height major every 2.5. You read a position
 * off it, the way you read a position off a tuner or a caliper.
 *
 * **It is not coloured.** A coloured track states a verdict at every point on
 * the axis including the ones the model said nothing about, and on a real
 * instrument the face is neutral and the *pointer* carries the state. So the
 * graduations are neutral, and hue lives on the three things that are actually
 * the reading — the needle, the readout and the tier name.
 *
 * The axis is the model's score (log-odds), not a percentage — see
 * src/lib/verdict.js for why. The tiers sit around the threshold, and the
 * uncertain band is drawn as a bracket under the engraving: neutral, because
 * it is a region of the axis, not a reading.
 *
 * Colour is never load-bearing alone: needle position, tier name and numeric
 * readout each carry the result independently.
 *
 * @param {number}  threshold  the decision threshold, in score units
 * @param {number}  bandLow    lower edge of the uncertain band, score units
 * @param {?number} score      the reading. Omit for an unmarked scale.
 * @param {boolean} labelled   draw numerals and tier names under the scale
 * @param {string}  height     Tailwind height for the graduation band
 */
export function VerdictScale({
  threshold,
  bandLow,
  score = null,
  labelled = true,
  height = 'h-12',
}) {
  const tiers = tiersFor(threshold, bandLow);
  const hasReading = score != null;
  const tier = hasReading ? tierFor(score, tiers) : null;
  const t = position(threshold);
  const s = hasReading ? position(score) : 0;
  const bandFrom = position(tiers[1].from);
  const bandTo = position(tiers[2].to);

  /* Park the needle at the left end for one frame, then release it, so the
     sweep runs on arrival instead of being skipped as the initial style. */
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setArmed(true));
    return () => cancelAnimationFrame(id);
  }, []);

  /* Centred on its point, except near the ends, where a centred label would
     hang off the panel. */
  const anchor = (x) => `translateX(${x < 0.08 ? '0' : x > 0.92 ? '-100%' : '-50%'})`;

  return (
    <div>
      <div className="relative pt-7">
        <span
          className="tick-label pointer-events-none absolute top-0 whitespace-nowrap text-secondary"
          style={{ left: `${t * 100}%`, transform: anchor(t) }}
          aria-hidden="true"
        >
          Threshold <span className="tabular">{formatScore(threshold)}</span>
        </span>

        {/* The engraving. Two layers on the same neutral fill: majors full
            height, fines at half height and bottom-aligned, which is what makes
            it read as a ruler rather than a barcode. */}
        <div className={`relative ${height}`} aria-hidden="true">
          <div className="graduation-major absolute inset-0 bg-muted" />
          <div className="graduation-fine absolute inset-x-0 bottom-0 h-1/2 bg-line-strong" />
        </div>

        {/* Threshold tick. Overshoots the band at both ends so it reads as a
            marking on the scale rather than another reading on it. Anchored
            top-and-bottom rather than by height, so it tracks the band. */}
        <div
          className="pointer-events-none absolute bottom-[-0.375rem] top-[1.375rem] w-px bg-primary"
          style={{ left: `${t * 100}%` }}
          aria-hidden="true"
        />

        {hasReading && (
          /* The needle. A zero-width rail so the two parts can each centre
             themselves on the reading without compounding the translate that
             positions the group. This is the only element on the scale that
             carries hue — it is the only one that is the reading. */
          <div
            className="pointer-events-none absolute bottom-[-0.5rem] top-[1.25rem] w-0"
            style={{
              left: `${(armed ? s : 0) * 100}%`,
              transition: 'left var(--duration-sweep) var(--ease-needle)',
            }}
            aria-hidden="true"
          >
            <div
              className="absolute inset-y-0 left-0 w-[3px] -translate-x-1/2 rounded-full ring-2 ring-[var(--panel)]"
              style={{ background: tier.token }}
            />
            <div
              className="absolute -top-[5px] left-0 h-2.5 w-2.5 -translate-x-1/2 rotate-45 ring-2 ring-[var(--panel)]"
              style={{ background: tier.token }}
            />
          </div>
        )}
      </div>

      {/* The uncertain band: a bracket under the engraving from the band's low
          edge to the top of the flagged tier, with the threshold inside it.
          A region of the axis, so neutral — the needle alone carries hue. */}
      <div className="relative mt-3 h-2" aria-hidden="true">
        <div
          className="absolute top-0 h-2 border-x border-b border-line-strong"
          style={{ left: `${bandFrom * 100}%`, width: `${(bandTo - bandFrom) * 100}%` }}
        />
      </div>

      {labelled && (
        <>
          {/* Numerals on the majors that carry them, so a position can be read
              as a number without the readout. */}
          <div className="relative mt-1 h-4" aria-hidden="true">
            {SCALE_NUMERALS.map((n) => {
              const x = position(n);
              return (
                <span
                  key={n}
                  className="tick-label tabular absolute top-0"
                  /* Always centred: a numeral names the tick under it, and
                     the outermost ones sit far enough in to have room. */
                  style={{ left: `${x * 100}%`, transform: 'translateX(-50%)' }}
                >
                  {formatScore(n, 0)}
                </span>
              );
            })}
          </div>

          {/* Tier names, each centred on the part of its tier that is on the
              scale. They are band labels, not boundary labels. */}
          <div className="relative mt-2 h-4" aria-hidden="true">
            {tiers.map((band) => {
              const from = position(band.from);
              const to = position(band.to);
              const mid = (from + to) / 2;
              /* A tier with only a sliver on the scale has no room for its
                 name at phone width; the headline and the bands table name
                 it there instead. */
              const narrow = to - from < 0.12;
              return (
                <span
                  key={band.id}
                  className={`tick-label absolute top-0 whitespace-nowrap ${narrow ? 'max-sm:hidden' : ''}`}
                  style={{
                    left: `${mid * 100}%`,
                    transform: anchor(mid),
                    color:
                      hasReading && tier.id === band.id
                        ? 'var(--text-primary)'
                        : undefined,
                  }}
                >
                  {band.label}
                </span>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
