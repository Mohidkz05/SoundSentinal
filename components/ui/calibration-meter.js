'use client';

import React from 'react';
import { tierFor, formatProbability } from '../../src/lib/verdict';
import { VerdictScale } from './verdict-scale';

/**
 * The Calibration Meter — the signature element.
 *
 * Every competitor renders a verdict. This renders a *reading*, against the
 * model's own published decision threshold, on a graduated scale that
 * desaturates where the model is least sure. Given the detector is a small CNN
 * that has not yet been trained to a competitive EER, a confident red "FAKE"
 * badge would be a lie. A meter that shows its own cutoff is not.
 *
 * The scale itself lives in `VerdictScale`, shared with the home page
 * illustration so the two can't drift.
 *
 * The meter is given the wide column on /result deliberately: on a graduated
 * scale, width *is* resolution — a longer scale puts more distance between two
 * readings a percentage point apart, so the reading gets easier to place. That
 * makes it the one element on the page that should absorb a large screen, while
 * the prose beside it stays at a readable measure.
 *
 * @param {number} probability  spoof_probability from the API, 0–1
 * @param {number} threshold    the model's calibrated operating point, 0–1
 */
export function CalibrationMeter({ probability = 0, threshold = 0.5 }) {
  const p = Math.min(Math.max(probability, 0), 1);
  const t = Math.min(Math.max(threshold, 0), 1);
  const tier = tierFor(p);
  const clears = p >= t;

  return (
    <figure className="w-full">
      <figcaption className="sr-only">
        Synthetic speech likelihood: {formatProbability(p)}, {tier.label}. The
        model&apos;s decision threshold is {formatProbability(t)}.
      </figcaption>

      {/* Readout. The tier name and the side of the threshold sit under the
          number as its own caption, rather than opposite it — at these sizes a
          justified pair reads as two competing headlines. */}
      <p className="tick-label">Synthetic likelihood</p>
      <div className="mt-3 flex flex-wrap items-end gap-x-8 gap-y-4">
        <output
          data-readout
          className="block text-readout font-medium leading-none tracking-[-0.04em]"
          style={{ color: tier.token }}
        >
          {formatProbability(p)}
        </output>

        <div className="flex flex-col gap-1 pb-1">
          <p
            className="text-h3 font-bold leading-none"
            style={{ color: tier.token, fontVariationSettings: '"wdth" 112' }}
          >
            {tier.label}
          </p>
          <p className="tick-label">
            {clears ? 'Above' : 'Below'} the decision threshold
          </p>
        </div>
      </div>

      <div className="mt-12">
        <VerdictScale probability={p} threshold={t} />
      </div>

      <p className="mt-8 max-w-[62ch] text-small text-secondary">
        {tier.detail}
      </p>
    </figure>
  );
}
