'use client';

import React from 'react';
import { setMotionPaused, useMotionPaused } from './three/use-stage';

/**
 * The visitor's switch for the background animation (WCAG 2.2.2: motion that
 * starts by itself and runs past five seconds needs a way to stop it). A
 * text control in the footer since 7 October 2026 — the owner did not want
 * it in the header. Remembered across visits; see use-stage.js.
 */
export function MotionSwitch() {
  const paused = useMotionPaused();
  return (
    <button
      type="button"
      onClick={() => setMotionPaused(!paused)}
      className="flex min-h-[var(--hit)] items-center text-small text-secondary hover:text-accent"
    >
      {paused ? 'Play background animation' : 'Pause background animation'}
    </button>
  );
}
