/**
 * Motion tokens and variants.
 *
 * These mirror the `--ease-*` and `--duration-*` custom properties in
 * globals.css one-for-one. CSS transitions and Framer Motion animations have to
 * agree, or a button that hovers in CSS and mounts in Motion will feel like two
 * different products.
 */

export const easing = {
  /** Decisive, no bounce. Every ordinary UI transition. */
  instrument: [0.2, 0.8, 0.2, 1],
  /** Overshoots ~4% and settles, the way a real meter needle does.
   *  Reserved for the reading itself — using it anywhere else spends the
   *  gesture and it stops meaning "a measurement landed". */
  needle: [0.34, 1.28, 0.44, 1],
  exit: [0.4, 0, 1, 1],
};

export const duration = {
  tap: 0.1,
  fast: 0.16,
  base: 0.24,
  slow: 0.42,
  sweep: 0.9,
};

/** Page and section entrances. Children stagger in reading order. */
export const stagger = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.06, delayChildren: 0.04 },
  },
};

export const riseIn = {
  hidden: { opacity: 0, y: 12 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: duration.slow, ease: easing.instrument },
  },
};

/** The dropzone reacting to a dragged file. Scale stays subtle — the border and
 *  fill do the talking, so the layout never shifts under the cursor. */
export const dropzone = {
  idle: { scale: 1 },
  active: {
    scale: 1.01,
    transition: { duration: duration.fast, ease: easing.instrument },
  },
};

/**
 * The needle sweep. Travels from zero to the reading once, on arrival.
 *
 * `custom` is the spoof probability, 0–1.
 */
export const needle = {
  hidden: { left: '0%' },
  show: (probability) => ({
    left: `${Math.min(Math.max(probability, 0), 1) * 100}%`,
    transition: { duration: duration.sweep, ease: easing.needle },
  }),
};

/** Reduced-motion fallback: keep the fade, drop the travel. Pair with
 *  Motion's `useReducedMotion()` at the call site. */
export const riseInStill = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: duration.base } },
};
