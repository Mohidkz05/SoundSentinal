import React from 'react';

/**
 * One measured value: a mono tick label, the value in tabular figures, and an
 * optional note on what was done with it. Used for the clip readouts on
 * /upload and /result, so a duration looks the same on both screens.
 */
export function Stat({ label, value, note }) {
  return (
    <div className="flex flex-col gap-[var(--space-tight)]">
      <p className="tick-label">{label}</p>
      <p className="tabular text-h3 leading-none text-primary">{value}</p>
      {note && <p className="text-small text-muted">{note}</p>}
    </div>
  );
}
