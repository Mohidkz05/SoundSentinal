'use client';

import dynamic from 'next/dynamic';
import React from 'react';
import { FlatMark } from '../ui/mark';

/**
 * The 3D layer's entry point. Import from here, never from the scene files.
 *
 * three.js and the renderer are around 250 kB of the bundle — more than the
 * rest of the application put together. Loading that before first paint would
 * make an ambient layer decide how fast the interface appears, which is exactly
 * backwards: the DOM is the product and the geometry is an enhancement of it.
 *
 * Each scene is split out behind `ssr: false`, so the pages ship and render
 * without three.js and the canvases arrive afterwards. Combined with the
 * WebGL check inside `Stage`, that means the 3D layer has two independent ways
 * to be absent and neither of them costs a user anything they needed.
 */

export const AmbientDepth = dynamic(
  () => import('./ambient-depth').then((m) => m.AmbientDepth),
  { ssr: false }
);

export const SpectralField = dynamic(
  () => import('./spectral-field').then((m) => m.SpectralField),
  { ssr: false }
);

export const WaveformDisplay = dynamic(
  () => import('./waveform-display').then((m) => m.WaveformDisplay),
  { ssr: false }
);

export const IntakeField = dynamic(
  () => import('./intake-field').then((m) => m.IntakeField),
  { ssr: false }
);

export const UncertaintyField = dynamic(
  () => import('./uncertainty-field').then((m) => m.UncertaintyField),
  { ssr: false }
);

/* The only one with a placeholder: the mark is brand furniture in a sticky
   header, so it has to be present on the very first frame rather than popping
   in. The flat mark holds its exact place until the geometry is ready. */
export const SignalMark = dynamic(
  () => import('./signal-mark').then((m) => m.SignalMark),
  {
    ssr: false,
    loading: () => <FlatMark className="h-7 w-7 shrink-0 p-0.5 text-accent" />,
  }
);
