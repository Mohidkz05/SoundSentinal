/**
 * Turning the model's score into something a person can read.
 *
 * THE SCALE IS THE MODEL'S SCORE, NOT A PERCENTAGE. The API returns
 * `spoof_score`, the log-odds of spoof (the difference of the two logits), and
 * that is what the page draws. It used to draw P(spoof) on 0–100%, which broke
 * the moment an SSL model was served: it pushes P to within a hair of 0 or 1,
 * and its threshold is P = 0.99966 (RESULTS.md Finding 12). On a percentage
 * scale the threshold and every high reading shared the last pixel, and a
 * real clip at P = 0.995 was headlined "Very likely AI generated" while the
 * model, below its threshold, called it real. The log-odds is also what every
 * threshold in this project was fitted on, so the page and the model now
 * read the same number.
 *
 * THE TIERS ARE PLACED AROUND THE THRESHOLD. They used to be fixed quarters
 * of the probability range, and because of that no tier text was allowed to
 * say which side of the threshold it was on. Now they are defined by it:
 *
 *   clear       below the uncertain band
 *   uncertain   the band up to the threshold — not flagged, but where genuine
 *               recordings sometimes score
 *   flagged     from the threshold up by the band's width
 *   strong      beyond that
 *
 * The band's lower edge comes from the API (`uncertain_band.low`): the 95th
 * percentile of held-out real speech's scores, stored in the checkpoint by
 * calibrate.py. The flagged tier mirrors that width above the line, which is
 * a convention, not a measurement — there is no held-out fake set to fit it
 * on — and the page says so. Hue says which side of the threshold (violet
 * below, orange above); the two middle tiers are the desaturated tokens.
 */

/** The drawn range, in score units. Covers every threshold served so far
 *  (dev-EER thresholds near −6, the served one at +8) with room either side;
 *  readings beyond it pin to the end and the readout still gives the number.
 *
 *  The odd-looking ends are chosen for the engraving in globals.css, which
 *  centres 50 fine ticks and 10 majors in their cells. Over 25 units starting
 *  at −11.25, every fine tick lands on a half unit and every major on a
 *  multiple of 2.5 (−10, −7.5 … +12.5), so the scale can carry numerals
 *  without changing the masks. */
export const SCALE = { min: -11.25, max: 13.75 };

/** Numerals drawn under the scale, on majors. */
export const SCALE_NUMERALS = [-10, -5, 0, 5, 10];

/** Band width used only when a checkpoint carries no calibration data. */
const FALLBACK_BAND = 4;

const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

/** P(spoof) -> log-odds. Only for readings from a server that predates
 *  `spoof_score`; clamped, so a saturated 1.0 becomes a large finite score. */
export function logit(p) {
  const q = Math.min(Math.max(Number(p) || 0, 1e-12), 1 - 1e-12);
  return Math.log(q) - Math.log1p(-q);
}

/**
 * Everything a reading is interpreted against, from an API response.
 * @returns {{score: ?number, threshold: number, bandLow: number,
 *            bandMeasured: boolean, bandSource: ?string}}
 */
export function readingScale(reading) {
  const threshold =
    typeof reading?.threshold_score === 'number'
      ? reading.threshold_score
      : logit(reading?.threshold ?? 0.5);
  const low = reading?.uncertain_band?.low;
  const bandMeasured = typeof low === 'number' && low < threshold;
  const score =
    typeof reading?.spoof_score === 'number'
      ? reading.spoof_score
      : typeof reading?.spoof_probability === 'number'
        ? logit(reading.spoof_probability)
        : null;
  return {
    score,
    threshold,
    bandLow: bandMeasured ? low : threshold - FALLBACK_BAND,
    bandMeasured,
    bandSource: bandMeasured ? reading.uncertain_band.source ?? null : null,
  };
}

/** The four tiers for a given threshold and band, low to high. */
export function tiersFor(threshold, bandLow) {
  const width = threshold - bandLow;
  return [
    {
      id: 'unlikely',
      from: -Infinity,
      to: bandLow,
      label: 'Clear',
      headline: 'Consistent with real speech',
      detail:
        'Below the uncertain band. Most genuine recordings score here, and nothing in this clip looked to the model like the synthetic speech it was trained on.',
      token: 'var(--verdict-unlikely)',
    },
    {
      id: 'possibly',
      from: bandLow,
      to: threshold,
      label: 'Uncertain',
      headline: 'Not flagged, but uncertain',
      detail:
        'Below the threshold, so the model does not flag it, but in the top range genuine recordings reach. Background noise, phone lines and heavy compression put real speech here.',
      token: 'var(--verdict-possibly)',
    },
    {
      id: 'likely',
      from: threshold,
      to: threshold + width,
      label: 'Flagged',
      headline: 'Flagged as likely AI generated',
      detail:
        'Above the threshold, so the model flags it, but close to the line. That is weaker evidence than a reading far past it.',
      token: 'var(--verdict-likely)',
    },
    {
      id: 'veryLikely',
      from: threshold + width,
      to: Infinity,
      label: 'Strong',
      headline: 'Very likely AI generated',
      detail:
        'Well past the threshold, further above it than the uncertain band reaches below it.',
      token: 'var(--verdict-verylikely)',
    },
  ];
}

/** The tier a score falls in. The threshold itself counts as flagged, as the
 *  server decides (`score >= threshold`). */
export function tierFor(score, tiers) {
  return tiers.find((t) => score < t.to) ?? tiers[tiers.length - 1];
}

/** 0–1 position of a score along the drawn scale, pinned at the ends. */
export function position(score) {
  return clamp01((score - SCALE.min) / (SCALE.max - SCALE.min));
}

/** Signed score for a readout: "+5.2", "−3.1". A true minus sign, so the
 *  column of tabular figures lines up and screen readers say "minus". */
export function formatScore(score, digits = 1) {
  const s = Number(score) || 0;
  const text = Math.abs(s).toFixed(digits);
  return Number(text) === 0 ? text : `${s < 0 ? '−' : '+'}${text}`;
}

/**
 * How sure the model is of the side of the line it chose, 0.5–1.
 *
 * The score re-centred on the threshold and put back through the sigmoid:
 * sigmoid(|score − threshold|). At the line it is 50%; each unit of score
 * beyond it is a factor of e in the odds. This is P(spoof) with the model's
 * prior shifted so that the threshold, not P = 0.5, is the even point — which
 * is what makes it usable. Raw P(spoof) is 0.999 at this threshold, so it
 * would call clips "99.9% fake" that the model calls real.
 *
 * It is the model's certainty, NOT a measured accuracy: nothing was fitted to
 * make 90% mean right nine times in ten. The page shows the measured error
 * rates beside it for that, and says so.
 */
export function confidence(score, threshold) {
  return 1 / (1 + Math.exp(-Math.abs(score - threshold)));
}

/** "97%"; near-certain readings keep a decimal, and never claim 100%. */
export function formatConfidence(c) {
  if (c >= 0.999) return '>99.9%';
  if (c >= 0.99) return `${(Math.floor(c * 1000) / 10).toFixed(1)}%`;
  return `${Math.floor(c * 100)}%`;
}

/** Percent string for a rate (EER, error rates). One decimal: a rate
 *  measured on a few thousand clips is not precise to more. */
export function formatPercent(rate) {
  const p = clamp01(Number(rate) || 0);
  return `${(p * 100).toFixed(1)}%`;
}
