'use client';

import React from 'react';
import { useTheme } from './theme';
import { ThemeButton } from './theme-button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@components/ui/tooltip';

/**
 * The header's dark-mode switch with the tooltip that names its action
 * (DESIGN.md). Loaded lazily by the header, which renders the bare
 * `ThemeButton` until it arrives: Radix's tooltip brings a positioning engine
 * that was the largest thing in the shared bundle of every page, for a hint
 * nobody needs before the page has painted.
 */
export function ThemeToggle() {
  const { darkMode } = useTheme();
  /* A short delay so the tooltip does not flash while the pointer is just
     passing over. */
  return (
    <TooltipProvider delayDuration={250}>
      <Tooltip>
        <TooltipTrigger asChild>
          <ThemeButton />
        </TooltipTrigger>
        <TooltipContent side="bottom">{darkMode ? 'Switch to light' : 'Switch to dark'}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
