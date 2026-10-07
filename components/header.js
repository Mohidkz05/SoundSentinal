'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from './theme';
import { SignalMark } from './three/lazy';

/* The product's navigation, not the site map. /result is the second step of
   the analyse flow and empty when visited directly, so it is reached through
   the flow (the stepper links back to it); /design is a reference for whoever
   builds this, and lives in the footer. */
const NAV = [
  { href: '/upload', label: 'Analyse', match: ['/upload', '/result'] },
  { href: '/#how-it-works', label: 'How it works', match: [] },
];

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

export default function Header() {
  const { darkMode, toggleDarkMode } = useTheme();
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-line bg-canvas/85 backdrop-blur-md">
      <div className="shell flex h-16 items-center gap-3 sm:gap-6">
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
          <span className="wordmark hidden text-small sm:inline" aria-hidden="true">
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
                className={`flex min-h-[var(--hit)] items-center rounded-md px-3 text-small transition-colors duration-fast ease-instrument ${
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
          <button
            type="button"
            onClick={toggleDarkMode}
            aria-label="Dark mode"
            aria-pressed={darkMode}
            className={ICON_BUTTON}
          >
            <ThemeIcon />
          </button>
        </div>
      </div>
    </header>
  );
}
