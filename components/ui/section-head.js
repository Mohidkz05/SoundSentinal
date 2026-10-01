import React from 'react';

/**
 * A band's heading and its explanation, side by side at desktop width.
 *
 * At full bleed a heading with a measure-limited paragraph under it leaves
 * most of the row empty, because the paragraph is held to a reading measure
 * and the heading is not. Pairing them across the row uses the width without
 * a line of prose growing past ~75 characters. Below `lg` they stack.
 *
 * Every band heading in the product goes through this, so the heading column,
 * the gap and the distance to the band's content are the same everywhere.
 */
export function SectionHead({ id, title, children, aside, level = 2 }) {
  const Heading = `h${level}`;
  return (
    <div className="grid gap-x-14 gap-y-4 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
      <Heading id={id} className={`${level === 2 ? 'text-h2' : 'text-h3'} text-balance`}>
        {title}
      </Heading>
      {(children || aside) && (
        <div className="flex flex-col gap-[var(--space-stack)] lg:flex-row lg:items-start lg:justify-between lg:gap-10">
          {children && (
            <div className="flex max-w-[72ch] flex-col gap-[var(--space-stack)] text-small text-secondary">
              {children}
            </div>
          )}
          {aside && <div className="shrink-0">{aside}</div>}
        </div>
      )}
    </div>
  );
}
