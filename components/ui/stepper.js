import React from 'react';
import Link from 'next/link';

/**
 * Where you are in the two-step flow. Replaces a "Step 1 of 2" eyebrow: the
 * sequence is real information, so it gets a real structure — an ordered
 * list, the current step marked with aria-current, and the finished step a
 * link back.
 */
const STEPS = [
  { href: '/upload', label: 'Choose a clip' },
  { href: '/result', label: 'Read the result' },
];

export function Stepper({ current }) {
  return (
    <nav aria-label="Progress">
      <ol className="flex flex-wrap items-center gap-x-3 gap-y-2">
        {STEPS.map((step, i) => {
          const state = i < current ? 'done' : i === current ? 'current' : 'next';
          const marker = (
            <span
              className={`tabular grid h-6 w-6 place-items-center rounded-full border text-[0.75rem] ${
                state === 'current'
                  ? 'border-accent bg-accent text-accent-contrast'
                  : state === 'done'
                    ? 'border-accent text-accent'
                    : 'border-line-strong text-muted'
              }`}
              aria-hidden="true"
            >
              {i + 1}
            </span>
          );
          const label = (
            <span className={state === 'current' ? 'text-primary' : 'text-secondary'}>
              {step.label}
            </span>
          );
          return (
            <li key={step.href} className="flex items-center gap-3">
              {i > 0 && <span className="h-px w-6 bg-line-strong" aria-hidden="true" />}
              {state === 'done' ? (
                <Link
                  href={step.href}
                  className="flex min-h-[var(--hit)] items-center gap-2 text-small hover:text-accent"
                >
                  {marker}
                  {label}
                  <span className="sr-only"> (completed)</span>
                </Link>
              ) : (
                <span
                  className="flex min-h-[var(--hit)] items-center gap-2 text-small"
                  aria-current={state === 'current' ? 'step' : undefined}
                >
                  {marker}
                  {label}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
