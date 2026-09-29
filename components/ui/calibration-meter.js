'use client';

import React from 'react';
import { formatScore, tierFor, tiersFor } from '../../src/lib/verdict';
import { VerdictScale } from './verdict-scale';

/**
 * The Calibration Meter — the signature element.
 *
 * Every competitor renders a verdict. This renders a *reading*, against the
 * model's own decision threshold, on a graduated scale whose tiers desaturate
 * where the model is least sure. A confident red "FAKE" badge would claim more
 * than a score against a threshold can. A meter that shows its own cutoff and
 * its uncertain band does not.
 *
 * The scale itself lives in `VerdictScale`, shared with the home page
 * illustration and /result so they can't drift.
 *
 * @param {number} score      spoof_score from the API (log-odds)
 * @param {number} threshold  threshold_score from the API
 * @param {number} bandLow    uncertain_band.low from the API
 */
export function CalibrationMeter({ score = 0, threshold, bandLow }) {
  const tier = tierFor(score, tiersFor(threshold, bandLow));
  const clears = score >= threshold;

  return (
    <figure className="w-full">
      <figcaption className="sr-only">
        Model score {formatScore(score)}, {tier.label}. The decision threshold
        is {formatScore(threshold)}.
      </figcaption>

      {/* Readout. The tier name and the side of the threshold sit under the
          number as its own caption, rather than opposite it — at these sizes a
          justified pair reads as two competing headlines. */}
      <p className="tick-label">Model score</p>
      <div className="mt-3 flex flex-wrap items-end gap-x-8 gap-y-4">
        <output
          data-readout
          className="block text-readout font-medium leading-none tracking-[-0.04em]"
          style={{ color: tier.token }}
        >
          {formatScore(score)}
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
        <VerdictScale score={score} threshold={threshold} bandLow={bandLow} />
      </div>

      <p className="mt-8 max-w-[62ch] text-small text-secondary">
        {tier.detail}
      </p>
    </figure>
  );
}
