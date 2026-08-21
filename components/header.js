'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from './theme';
import { SignalMark } from './three/lazy';

const NAV = [
  { href: '/', label: 'Home' },
  { href: '/upload', label: 'Upload' },
  { href: '/result', label: 'Result' },
  { href: '/design', label: 'Design' },
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

export default function Header() {
  const { toggleDarkMode } = useTheme();
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-line bg-canvas/85 backdrop-blur-md">
      <div className="shell flex h-16 items-center gap-2 sm:gap-6">
        <Link
          href="/"
          className="flex items-center gap-2.5 rounded-sm text-primary"
          aria-label="SoundSentinal home"
        >
          <SignalMark />
          {/* The wordmark is what goes below `sm`, not the navigation. The
              header used to drop the nav entirely on a phone, which left the
              mark and the theme toggle and no way to reach /result or /design
              at all — a brand asset kept at the cost of the only navigation on
              the page. The mark still goes home, so nothing is lost. */}
          <span
            className="hidden text-small font-bold tracking-[0.02em] sm:inline"
            style={{ fontVariationSettings: '"wdth" 112' }}
          >
            SOUNDSENTINAL
          </span>
        </Link>

        <nav className="ml-auto flex items-center gap-0.5 sm:gap-1" aria-label="Main">
          {NAV.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? 'page' : undefined}
                className={`rounded-md px-2 py-2 text-small transition-colors duration-fast ease-instrument sm:px-3 ${
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

        <button
          type="button"
          onClick={toggleDarkMode}
          aria-label="Toggle dark mode"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line text-secondary transition-colors duration-fast ease-instrument hover:border-accent hover:text-accent"
        >
          <ThemeIcon />
        </button>
      </div>
    </header>
  );
}
