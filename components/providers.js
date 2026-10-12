'use client';

import React from 'react';
import { MotionConfig } from 'motion/react';
import { TooltipProvider } from '@components/ui/tooltip';

/**
 * App-wide client providers.
 *
 * MotionConfig reducedMotion="user": every Motion animation in the app drops
 * its movement (keeping opacity) when the OS asks for reduced motion — the
 * same rule globals.css applies to CSS animations. TooltipProvider is what
 * shadcn/ui's Tooltip needs once at the root; a short delay so tooltips do
 * not flash while the pointer is just passing over.
 */
export function Providers({ children }) {
  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={250}>{children}</TooltipProvider>
    </MotionConfig>
  );
}
