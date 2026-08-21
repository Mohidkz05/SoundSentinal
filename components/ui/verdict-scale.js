'use client';

import React, { useEffect, useState } from 'react';
import { TIERS, tierFor, formatProbability } from '../../src/lib/verdict';

/**
 * The graduated scale a reading is read off.
 *
 * Two decisions here, both arrived at by removing something.
 *
 * **It is not a filled bar.** A continuous fill reads as a progress bar — a
 * quantity accumulating towards completion — and a probability is not that. It
 * also drags the eye along its whole length when the only thing carrying the
 * result is one point on it. So the scale is engraved instead: fifty fine
 * graduations at two percentage points each, a full-height major every ten. You
 * read a position off it, the way you read a position off a tuner or a caliper.
 *
 * **It is not coloured.** The scale used to render the diverging violet↔orange
 * ramp along its length. That was wrong twice over: a coloured track states a
 * verdict at every point on the axis including the ones the model said nothing
 * about, and on a real instrument the face is neutral and the *pointer* carries
 * the state. So the graduations are neutral, and hue lives on the three things
 * that are actually the reading — the needle, the readout and the tier name.
 *
 * The colour rules survive that move intact, because they were always rules
 * about the reading rather than about the track: hue still says which side of
 * the threshold, and the middle tiers are the desaturated tokens, so a needle
 * near the threshold still drains towards grey.
 *
 * Colour is never load-bearing alone: needle position, tier name and numeric
 * readout each carry the result independently.
 *
 * @param {number}  threshold    the model's calibrated operating point, 0–1
 * @param {?number} probability  the reading, 0–1. Omit for an unmarked scale.
 * @param {boolean} labelled     draw the tier names under the bands
 * @param {string}  height       Tailwind height for the graduation band
 */
export function VerdictScale({
  threshold = 0.5,
  probability = null,
  labelled = true,
  height = 'h-12',
}) {
  const t = Math.min(Math.max(threshold, 0), 1);
  const hasReading = probability != null;
  const p = hasReading ? Math.min(Math.max(probability, 0), 1) : 0;
  const tier = hasReading ? tierFor(p) : null;

  /* Park the needle at zero for one frame, then release it, so the sweep runs
     on arrival instead of being skipped as the initial style. */
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setArmed(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div>
      <div className="relative pt-7">
        <span
          className="tick-label pointer-events-none absolute top-0 whitespace-nowrap text-secondary"
          style={{
            left: `${t * 100}%`,
            /* Centred on the tick, except near the ends — at a threshold of
               0.02 a centred label hangs off the panel edge. */
            transform: `translateX(${t < 0.12 ? '0' : t > 0.88 ? '-100%' : '-50%'})`,
          }}
          aria-hidden="true"
        >
          Threshold {formatProbability(t)}
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
              left: `${(armed ? p : 0) * 100}%`,
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

      {/* Band names, each centred over its own quarter. Spreading these edge to
          edge would put "Unlikely" at 0% and "Very likely" at 100% and so imply
          they were boundary labels. They are band labels. */}
      {labelled && (
        <div className="mt-3 flex" aria-hidden="true">
          {TIERS.map((band) => (
            <span
              key={band.id}
              className="tick-label flex-1 text-center"
              style={
                hasReading && tier.id === band.id
                  ? { color: 'var(--text-primary)' }
                  : undefined
              }
            >
              {band.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
