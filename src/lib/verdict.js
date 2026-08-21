/**
 * Turning a spoof probability into something a person can read.
 *
 * Two separate ideas share this screen and must not be confused:
 *
 *   - The **tiers** below are fixed quarters of the probability range. They
 *     describe the reading itself ("this clip scored 0.71"), which is why they
 *     don't move when the model is retrained.
 *   - The **threshold** is the model's calibrated operating point, computed
 *     from dev-set EER during training and returned by the API. It's the line
 *     the model itself uses to decide, and it moves between training runs.
 *
 * The meter draws both: the reading against the tier bands, and the threshold
 * as a labelled tick. Showing the threshold is the honest move — it makes the
 * cutoff visible instead of implying a verdict came from nowhere.
 *
 * Because they are independent, **no tier description may mention which side of
 * the threshold it falls on.** Two of them used to — "sits below the decision
 * threshold", "clears the threshold" — which was true only while the threshold
 * happened to be 0.5, the tier boundary they sat on. The first real checkpoint
 * served a calibrated 0.413 and the page immediately contradicted itself: a
 * reading of 48.7% labelled "Possibly", described as below the threshold, above
 * a line saying it was above it. The threshold relation is stated once, by the
 * page, from the value the API actually returned.
 */

export const TIERS = [
  {
    id: 'unlikely',
    max: 0.25,
    label: 'Unlikely',
    headline: 'Unlikely to be AI generated',
    detail: 'Nothing in this clip matched the synthesis artefacts the model looks for.',
    token: 'var(--verdict-unlikely)',
  },
  {
    id: 'possibly',
    max: 0.5,
    label: 'Possibly',
    headline: 'Possibly AI generated',
    detail: 'Some of what the model treats as synthesis artefacts is present, but weakly.',
    token: 'var(--verdict-possibly)',
  },
  {
    id: 'likely',
    max: 0.75,
    label: 'Likely',
    headline: 'Likely AI generated',
    detail: 'Artefacts the model associates with synthesis are present across much of the clip.',
    token: 'var(--verdict-likely)',
  },
  {
    id: 'veryLikely',
    max: 1.01,
    label: 'Very likely',
    headline: 'Very likely AI generated',
    detail: 'Strong synthesis artefacts across the clip.',
    token: 'var(--verdict-verylikely)',
  },
];

/** @param {number} probability 0–1 spoof probability from the API. */
export function tierFor(probability) {
  const p = Math.min(Math.max(Number(probability) || 0, 0), 1);
  return TIERS.find((t) => p < t.max) ?? TIERS[TIERS.length - 1];
}

/** Percent string for a readout. One decimal — the model is not precise enough
 *  to justify more, and two decimals imply confidence that isn't there. */
export function formatProbability(probability) {
  const p = Math.min(Math.max(Number(probability) || 0, 0), 1);
  return `${(p * 100).toFixed(1)}%`;
}
