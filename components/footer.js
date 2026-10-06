import React from 'react';
import Link from 'next/link';
import { FlatMark } from './ui/mark';

/**
 * One ending for every page. It carries the standing disclaimer — which used
 * to be written separately, and differently, at the foot of each page — the
 * links that are for builders rather than for the person checking a clip
 * (the design system), and the credits the training data's licences ask for.
 */
const LINKS = [
  { href: '/upload', label: 'Analyse a clip' },
  { href: '/#how-it-works', label: 'How it works' },
  { href: '/design', label: 'Design system' },
];

export default function Footer() {
  return (
    <footer className="band mt-auto">
      <div className="grid gap-[var(--space-group)] lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)_auto] lg:gap-x-14">
        <div className="flex flex-col gap-[var(--space-stack)]">
          <span className="flex items-center gap-2.5 text-primary">
            <FlatMark className="h-5 w-5 text-accent" />
            <span className="wordmark text-small">SOUNDSENTINAL</span>
          </span>
          <p className="text-small text-muted">
            A university research project. A reading is a signal worth following
            up, never proof that a recording is real or fake.
          </p>
        </div>

        <p className="max-w-[72ch] text-small text-muted">
          Trained on ASVspoof 2019 LA (ODC-By), SpeechFake (CC BY 4.0) and
          clips we generated with eight open text-to-speech models, with an
          XLS-R front-end (Apache 2.0). Clips you upload are processed in
          memory and not stored.
        </p>

        <nav aria-label="Footer" className="flex flex-col items-start gap-0.5 lg:items-end">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="flex min-h-[var(--hit)] items-center text-small text-secondary hover:text-accent"
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </footer>
  );
}
