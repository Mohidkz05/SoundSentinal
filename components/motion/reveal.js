'use client';

import React from 'react';
import * as m from 'motion/react-m';
import { duration, easing } from '../../src/lib/motion';

/**
 * The app's entrance motion, in three pieces, all on the tokens in
 * src/lib/motion.js so Motion and the CSS transitions agree.
 *
 *   <Rise>        above the fold: travels up into place but starts fully
 *                 opaque — content that begins at opacity 0 does not count as
 *                 painted, and that held /upload's LCP back by ~3 s
 *                 (Lighthouse, 7 October 2026).
 *   <Reveal>      below the fold: fades and rises in once, the first time a
 *                 fifth of it scrolls into view. Never replays.
 *   <Stagger> +   a list whose items arrive in reading order, 60 ms apart —
 *   <StaggerItem> the order is the information (steps, rows, caveats).
 *
 * MotionConfig reducedMotion="user" (components/providers.js) removes the
 * travel for anyone who asks for reduced motion; opacity still resolves.
 */

const rise = {
  hidden: { opacity: 1, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: duration.slow, ease: easing.instrument } },
};

const reveal = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: duration.slow, ease: easing.instrument } },
};

const group = (delay = 0) => ({
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: delay } },
});

const VIEW = { once: true, amount: 0.2 };

export function Rise({ as = 'div', delay = 0, className, children, ...props }) {
  const C = m[as];
  return (
    <C
      className={className}
      initial="hidden"
      animate="show"
      variants={{ ...rise, show: { ...rise.show, transition: { ...rise.show.transition, delay } } }}
      {...props}
    >
      {children}
    </C>
  );
}

export function Reveal({ as = 'div', className, children, ...props }) {
  const C = m[as];
  return (
    <C className={className} initial="hidden" whileInView="show" viewport={VIEW} variants={reveal} {...props}>
      {children}
    </C>
  );
}

/** `above` = the list is above the fold: items rise (opaque) on mount
 *  instead of waiting to be scrolled into view. */
export function Stagger({ as = 'div', above = false, delay = 0, className, children, ...props }) {
  const C = m[as];
  const trigger = above ? { animate: 'show' } : { whileInView: 'show', viewport: VIEW };
  return (
    <C className={className} initial="hidden" {...trigger} variants={group(delay)} {...props}>
      {children}
    </C>
  );
}

export function StaggerItem({ as = 'div', above = false, className, children, ...props }) {
  const C = m[as];
  return (
    <C className={className} variants={above ? rise : reveal} {...props}>
      {children}
    </C>
  );
}
