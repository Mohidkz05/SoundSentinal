'use client';

import React from 'react';
import { LazyMotion, MotionConfig } from 'framer-motion';

/**
 * App-wide client providers.
 *
 * LazyMotion: components use Motion's slim `m` elements, and the animation
 * features (~25 kB gzipped) arrive as their own chunk after first paint
 * instead of inside every page's first load. Entrances start as soon as it
 * lands; nothing waits on it to be readable (see components/motion/reveal.js).
 *
 * Import paths matter here. `motion/react` re-exports through
 * `import * as fm from 'framer-motion'` and reads `fm.motion`, which pins the
 * full-feature `motion` component into any bundle that touches it, LazyMotion
 * or not (motion 13.1). So `m` comes from `motion/react-m`, and LazyMotion,
 * MotionConfig and the features from `framer-motion` itself — the package
 * `motion` wraps, at the same version. Importing `motion/react` anywhere in
 * the app puts ~35 kB back on every page.
 *
 * MotionConfig reducedMotion="user": every Motion animation in the app drops
 * its movement (keeping opacity) when the OS asks for reduced motion — the
 * same rule globals.css applies to CSS animations.
 *
 * The header's tooltip carries its own provider (components/theme-toggle.js),
 * so Radix's tooltip and its positioning engine stay out of the root bundle.
 */
const loadFeatures = () => import('./motion/features').then((mod) => mod.default);

export function Providers({ children }) {
  return (
    <LazyMotion features={loadFeatures}>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </LazyMotion>
  );
}
