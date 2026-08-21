'use client';

import { useEffect, useState } from 'react';
import { Color } from 'three';
import { useTheme } from '../theme';

/**
 * The 3D layer's colours come from the design tokens, not from itself.
 *
 * This is the same rule the rest of the system follows — tokens are defined
 * once in `globals.css` — but it matters more here, because a hardcoded hex in
 * a shader is invisible to anyone auditing the palette and silently ignores the
 * theme. Reading the resolved custom properties means light/dark, and any
 * future change to a token, reach the geometry for free.
 *
 * `getComputedStyle` substitutes `var()` chains at computed-value time, so
 * `--accent` comes back as the literal hex of `--color-teal-700`, not as the
 * reference.
 */

const TOKENS = {
  canvas: '--canvas',
  panel: '--panel',
  raised: '--raised',
  line: '--line',
  lineStrong: '--line-strong',
  accent: '--accent',
  accentQuiet: '--accent-quiet',
  faint: '--text-faint',
  muted: '--text-muted',
  primary: '--text-primary',
  unlikely: '--verdict-unlikely',
  possibly: '--verdict-possibly',
  likely: '--verdict-likely',
  veryLikely: '--verdict-verylikely',
};

/* Light-theme values, used for the single render that happens before the first
   effect runs. Without these the geometry would flash black. */
const FALLBACK = {
  canvas: '#f7fbfb',
  panel: '#eff5f4',
  raised: '#ffffff',
  line: '#d7e0df',
  lineStrong: '#bfcac9',
  accent: '#056e6a',
  accentQuiet: '#b2e5e1',
  faint: '#7d8988',
  muted: '#5f6c6b',
  primary: '#142221',
  unlikely: '#683baf',
  possibly: '#6a549a',
  likely: '#ad5b32',
  veryLikely: '#bb4f05',
};

function read() {
  const style = getComputedStyle(document.documentElement);
  const out = {};
  for (const [key, prop] of Object.entries(TOKENS)) {
    const value = style.getPropertyValue(prop).trim();
    out[key] = new Color(value || FALLBACK[key]);
  }
  return out;
}

function fallbackPalette() {
  const out = {};
  for (const [key, value] of Object.entries(FALLBACK)) out[key] = new Color(value);
  return out;
}

/**
 * Returns a `THREE.Color` per semantic token, re-read whenever the theme flips.
 *
 * The returned object identity changes on a theme change, which is the signal
 * downstream effects use to push new values into shader uniforms.
 */
export function usePalette() {
  const { darkMode } = useTheme() ?? {};
  const [palette, setPalette] = useState(fallbackPalette);

  useEffect(() => {
    setPalette(read());
  }, [darkMode]);

  return palette;
}

/** Look up a verdict tier's colour by the tier id used in `src/lib/verdict.js`. */
export function tierColor(palette, tierId) {
  return palette[tierId] ?? palette.possibly;
}
