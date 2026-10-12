'use client';

import React, { useMemo } from 'react';
import { motion } from 'motion/react';
import { duration, easing } from '../../src/lib/motion';

/**
 * The analyse page's instrument: a timeline of the clip with the part the
 * model actually hears marked on it.
 *
 * The model scores the first WINDOW seconds and nothing else — the tool's
 * biggest limit, and one a visitor cannot guess. Drawing it as a lit window
 * on a ruler says it without a caveat: drop a clip and its waveform unrolls
 * along the ruler, and you can see which part is scored and which is not.
 *
 * The axis is linear across the window and logarithmic after it, out to the
 * two minutes a video is cut to. Linear all the way, the 4-second window would
 * be 3% of the width; the compressed tail is labelled as such under the ruler.
 *
 * Entirely visual (aria-hidden): the page states the same fact in text.
 */

const WINDOW = 4;          // seconds the model reads (MODEL_WINDOW_SECONDS)
const MAX = 120;           // the ruler's end: video is cut to two minutes
const W = 0.26;            // share of the width the window takes
const COLUMNS = 168;

/** Seconds → position across the ruler, 0..1. */
function x(t) {
  if (t <= WINDOW) return (t / WINDOW) * W;
  return W + (Math.log(t / WINDOW) / Math.log(MAX / WINDOW)) * (1 - W);
}

/** Position 0..1 → seconds. */
function t(pos) {
  if (pos <= W) return (pos / W) * WINDOW;
  return WINDOW * Math.exp(((pos - W) / (1 - W)) * Math.log(MAX / WINDOW));
}

const MAJOR = [
  [0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4 s'],
  [10, '10 s'], [20, '20 s'], [30, '30 s'], [60, '1 min'], [120, '2 min'],
];
/* Labels that survive a phone: the window's edges and the ruler's end. */
const NARROW_KEEP = new Set([0, 4, 30, 120]);
const MINOR = [0.5, 1.5, 2.5, 3.5, 5, 6, 7, 8, 9, 15, 40, 50, 90];

/* A speech-like idle envelope — syllables gated into words — so the empty
   ruler already looks like what it is for. */
function idle(i) {
  const u = i / COLUMNS;
  const syllables = Math.abs(Math.sin(u * 47)) ** 1.5;
  const words = Math.max(0, Math.sin(u * 19 + 0.4) + 0.3) ** 0.6;
  const grain = 0.75 + 0.25 * Math.abs((Math.sin(i * 12.9898) * 43758.5453) % 1);
  // Rounded: server and browser maths can differ in the last digit, and an
  // unrounded height made the server HTML fail to hydrate.
  return Math.round((0.06 + 0.86 * syllables * words * grain) * 1000) / 1000;
}

/**
 * `live`: the envelope is growing in real time (the microphone). Bars follow
 * it without the unroll delay, and the end marker becomes a plain cursor.
 * `dim`: nothing is loaded and a prompt sits on top, so the idle envelope
 * fades to texture — at full strength it read as a clip already loaded.
 */
export function ListeningWindow({ peaks = null, duration = null, active = false, live = false, dim = false }) {
  const hasClip = Boolean(peaks?.length && duration > 0);

  const bars = useMemo(() => {
    return Array.from({ length: COLUMNS }, (_, i) => {
      const from = t(i / COLUMNS);
      const to = t((i + 1) / COLUMNS);
      const inWindow = to <= WINDOW + 1e-9;
      if (!hasClip) return { v: idle(i), inWindow, present: true };
      if (from >= duration) return { v: 0, inWindow, present: false };
      // Loudest bucket this column spans, so short clips keep their shape.
      const a = Math.floor((from / duration) * peaks.length);
      const b = Math.max(a + 1, Math.ceil((Math.min(to, duration) / duration) * peaks.length));
      let m = 0;
      for (let j = a; j < Math.min(b, peaks.length); j++) m = Math.max(m, peaks[j]);
      return { v: Math.round(Math.sqrt(m) * 1000) / 1000, inWindow, present: true };
    });
  }, [hasClip, peaks, duration]);

  const end = hasClip ? Math.min(duration, MAX) : null;

  return (
    <div className="relative select-none" aria-hidden="true">
      {/* The window: lit, labelled, and the brightest thing on the page. */}
      <div
        className={`absolute bottom-[3.25rem] left-0 top-0 rounded-[var(--radius-lg)] border-[1.5px] border-accent bg-accent-wash transition-[box-shadow] duration-[var(--duration-base)] ease-[var(--ease-instrument)] ${
          active ? 'shadow-[0_0_0_6px_var(--accent-quiet)]' : ''
        }`}
        style={{ width: `${W * 100}%` }}
      >
        <div className="absolute inset-x-0 top-0 flex items-baseline justify-between gap-2 px-3 pt-2.5 sm:px-4 sm:pt-3">
          <span className="text-small font-semibold text-accent">
            <span className="sm:hidden">Heard</span>
            <span className="hidden sm:inline">The model listens here</span>
          </span>
          <span className="tabular hidden text-small text-accent md:inline">0–4 s</span>
        </div>
      </div>

      {/* Bars, on the same axis as the ruler. */}
      <div className="relative flex h-[clamp(10rem,24vh,15rem)] lg:h-[clamp(16rem,32vh,19rem)] items-end gap-[2px] px-[2px] pb-2 pt-16">
        {bars.map((bar, i) => (
          <span
            key={i}
            className={`flex-1 rounded-full ${
              live
                ? 'transition-[height] duration-[60ms] ease-linear'
                : 'transition-[height,opacity,background-color] duration-[var(--duration-slow)] ease-[var(--ease-instrument)]'
            } ${
              bar.inWindow ? 'bg-accent' : 'bg-line-strong'
            }`}
            style={{
              /* A chosen clip unrolls along the ruler left to right, the way
                 it plays; the idle envelope just settles. */
              transitionDelay: hasClip && !live ? `${i * 3}ms` : '0ms',
              height: bar.present ? `${Math.max(bar.v, 0.035) * (active ? 100 : 88)}%` : '2px',
              opacity: !bar.present
                ? 0.35
                : hasClip
                  ? bar.inWindow ? 1 : 0.75
                  : dim
                    ? bar.inWindow ? 0.28 : 0.13
                    : bar.inWindow ? 0.7 : 0.45,
            }}
          />
        ))}

        {live && end !== null && (
          <div
            className="pointer-events-none absolute bottom-2 top-16 w-0.5 rounded-full bg-accent"
            style={{ left: `${x(end) * 100}%` }}
          />
        )}

        {!live && end !== null && end > 0.2 && (
          <motion.div
            key={end}
            className="pointer-events-none absolute bottom-2 top-16 w-px bg-primary"
            style={{ left: `${x(end) * 100}%` }}
            initial={{ opacity: 0, scaleY: 0.4 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ duration: duration.slow, ease: easing.instrument, delay: 0.35 }}
          >
            <span className="tick-label absolute -top-5 left-1.5 whitespace-nowrap text-primary">
              {duration > MAX ? 'cut at 2 min' : `end · ${duration < 10 ? duration.toFixed(1) : Math.round(duration)} s`}
            </span>
          </motion.div>
        )}
      </div>

      {/* The ruler. */}
      <div className="relative h-[3.25rem] border-t border-line-strong">
        {MINOR.map((s) => (
          <span key={s} className="absolute top-0 h-1.5 w-px bg-line-strong" style={{ left: `${x(s) * 100}%` }} />
        ))}
        {MAJOR.map(([s, label]) => (
          <span key={s} className="absolute top-0" style={{ left: `${x(s) * 100}%` }}>
            <span className={`block h-3 w-px ${s === WINDOW ? 'bg-accent' : 'bg-line-strong'}`} />
            <span
              className={`tick-label tabular absolute top-4 whitespace-nowrap ${s === 0 ? 'left-0' : s === MAX ? 'right-0' : '-translate-x-1/2'} ${
                NARROW_KEEP.has(s) ? '' : 'max-sm:hidden'
              } ${s === WINDOW ? 'text-accent' : ''}`}
            >
              {label}
            </span>
          </span>
        ))}
        <span className="absolute bottom-0 right-0 text-[0.75rem] text-muted max-sm:hidden">
          compressed after 4 s
        </span>
      </div>
    </div>
  );
}
