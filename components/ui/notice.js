import React from 'react';

/**
 * A sentence the user has to notice: something failed, or a caveat that
 * changes how to read what is next to it.
 *
 *   tone="danger"  a system failure (rejected file, server down). role=alert,
 *                  the danger token — never a verdict orange.
 *   tone="caveat"  a qualification of a result or preview. Muted text and a
 *                  hairline mark; it must not look like an error.
 *
 * Icon plus text, never colour alone. Errors say what happened and what to do.
 */
const ICONS = {
  danger: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7.5v5M12 16.2v.1" />
    </>
  ),
  caveat: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5.5M12 7.8v.1" />
    </>
  ),
};

export function Notice({ tone = 'caveat', children, className = '' }) {
  const danger = tone === 'danger';
  return (
    <p
      role={danger ? 'alert' : undefined}
      className={`flex max-w-[72ch] items-start gap-2.5 text-small ${
        danger ? 'text-danger' : 'text-muted'
      } ${className}`}
    >
      <svg
        viewBox="0 0 24 24"
        className="mt-[0.2em] h-4 w-4 flex-none"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        aria-hidden="true"
      >
        {ICONS[tone]}
      </svg>
      <span>{children}</span>
    </p>
  );
}
