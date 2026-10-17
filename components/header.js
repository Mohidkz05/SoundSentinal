'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import dynamic from 'next/dynamic';
import { SignalMark } from './three/lazy';
import { ThemeButton } from './theme-button';

/* The product's navigation, not the site map. /result is the second step of
   the analyse flow and empty when visited directly, so it is reached through
   the flow (the stepper links back to it); /design is a reference for whoever
   builds this, and lives in the footer. */
const NAV = [
  { href: '/upload', label: 'Analyse', match: ['/upload', '/result'] },
  { href: '/#how-it-works', label: 'How it works', match: [] },
];

/* The tooltip arrives after first paint; until then the plain button holds its
   exact place and already works. See components/theme-toggle.js. */
const ThemeToggle = dynamic(() => import('./theme-toggle').then((m) => m.ThemeToggle), {
  ssr: false,
  loading: () => <ThemeButton />,
});

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-line bg-canvas/85 backdrop-blur-md">
      <div className="shell flex h-[4.5rem] items-center gap-3 sm:gap-6">
        <Link
          href="/"
          className="flex min-h-[var(--hit)] items-center gap-2.5 rounded-sm text-primary"
          aria-label="SoundSentinal home"
        >
          <SignalMark />
          {/* Below `sm` the mark stands alone: at 390px the wordmark, two
              nav links and a 44px toggle do not fit on one row, and the nav is
              the thing a phone user can't do without. The mark still goes home
              and the footer carries the full wordmark. */}
          <span className="wordmark hidden text-lead sm:inline" aria-hidden="true">
            SOUNDSENTINAL
          </span>
        </Link>

        <nav className="ml-auto flex items-center gap-1" aria-label="Main">
          {NAV.map(({ href, label, match }) => {
            const active = match.includes(pathname);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? 'page' : undefined}
                className={`flex min-h-[var(--hit)] items-center rounded-md px-3 text-body font-medium transition-colors duration-fast ease-instrument ${
                  active
                    ? 'bg-overlay text-primary'
                    : 'text-secondary hover:bg-overlay hover:text-primary'
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* The pause-animation switch moved to the footer (7 October 2026):
            the header keeps only the theme toggle. */}
        <div className="flex shrink-0 items-center">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
