'use client';

import React from 'react';
import { useTheme } from './theme';

/**
 * The dark-mode control itself, without its tooltip. The header renders it on
 * the very first frame; components/theme-toggle.js wraps it in the tooltip
 * once that has loaded.
 */

/* Both icons render every time and the `dark:` variant picks one. Deriving
   visibility from CSS rather than from `darkMode` state avoids a hydration
   mismatch — the class is on <html> before React boots, but the state isn't
   populated until an effect runs. */
function ThemeIcon() {
  return (
    <>
      <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] dark:hidden" fill="none"
           stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="4.2" />
        <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4L17 7M7 17l-1.6 1.6" />
      </svg>
      <svg viewBox="0 0 24 24" className="hidden h-[18px] w-[18px] dark:block" fill="currentColor"
           aria-hidden="true">
        <path d="M20.3 14.9A8.5 8.5 0 1 1 9.1 3.7a7 7 0 0 0 11.2 11.2z" />
      </svg>
    </>
  );
}

const ICON_BUTTON =
  'grid h-[var(--hit)] w-[var(--hit)] shrink-0 place-items-center rounded-md border border-line text-secondary transition-colors duration-fast ease-instrument hover:border-accent hover:text-accent';

/* Takes the props a Radix trigger passes through, and composes its click
   handler with its own rather than letting one replace the other. */
export const ThemeButton = React.forwardRef(function ThemeButton({ onClick, ...props }, ref) {
  const { darkMode, toggleDarkMode } = useTheme();
  return (
    <button
      ref={ref}
      type="button"
      {...props}
      onClick={(e) => {
        onClick?.(e);
        toggleDarkMode();
      }}
      aria-label="Dark mode"
      aria-pressed={darkMode}
      className={ICON_BUTTON}
    >
      <ThemeIcon />
    </button>
  );
});
