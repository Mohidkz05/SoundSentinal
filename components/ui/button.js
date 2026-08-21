'use client';

import React from 'react';

/**
 * Buttons.
 *
 * Four variants, and the rule is that a screen gets one `primary` at most —
 * if two things are primary, neither is. `quiet` is the default for anything
 * that isn't the point of the screen.
 *
 * Press feedback is a 1px downward nudge rather than a scale, so text stays on
 * the pixel grid and doesn't soften mid-press.
 */

const base = [
  'inline-flex items-center justify-center gap-2 select-none',
  'font-sans font-semibold whitespace-nowrap',
  'rounded-[var(--radius-md)] border',
  'transition-[background-color,border-color,color,box-shadow,translate]',
  'duration-[var(--duration-fast)] ease-[var(--ease-instrument)]',
  'active:translate-y-px',
  'disabled:pointer-events-none disabled:opacity-45',
].join(' ');

const variants = {
  // The one action that moves the user forward.
  primary: [
    'bg-[var(--accent)] text-[var(--accent-contrast)] border-transparent',
    'shadow-[var(--shadow-panel)]',
    'hover:bg-[var(--accent-hover)]',
  ].join(' '),

  // Equal-weight alternatives, and anything on a panel.
  secondary: [
    'bg-[var(--raised)] text-[var(--text-primary)] border-[var(--line-strong)]',
    'shadow-[var(--shadow-panel)]',
    'hover:border-[var(--accent)] hover:text-[var(--accent)]',
  ].join(' '),

  // Tertiary actions: back, cancel, change file.
  quiet: [
    'bg-transparent text-[var(--text-secondary)] border-transparent',
    'hover:bg-[var(--overlay)] hover:text-[var(--text-primary)]',
  ].join(' '),

  // Destructive or dismissive. Uses the danger token, never the verdict oranges.
  danger: [
    'bg-transparent text-[var(--danger)] border-[var(--danger)]',
    'hover:bg-[var(--danger)] hover:text-[var(--canvas)]',
  ].join(' '),
};

const sizes = {
  sm: 'h-9 px-3.5 text-[0.8125rem]',
  md: 'h-11 px-5 text-[0.9375rem]',
  lg: 'h-[3.25rem] px-7 text-base',
};

export function Button({
  variant = 'secondary',
  size = 'md',
  className = '',
  loading = false,
  children,
  ...props
}) {
  return (
    <button
      {...props}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      aria-busy={loading || undefined}
      /* After the spread: a loading button is disabled whatever else was
         passed, so the click cannot land twice. */
      disabled={loading || props.disabled}
    >
      {loading && <Spinner />}
      {children}
    </button>
  );
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path
        d="M14.5 8A6.5 6.5 0 0 0 8 1.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}
