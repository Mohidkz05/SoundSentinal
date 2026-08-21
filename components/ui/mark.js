import React from 'react';

/**
 * The flat wordmark — four bars at the brand hue, a spectrogram slice.
 *
 * Lives apart from its 3D counterpart on purpose: this file pulls in no
 * three.js, so it can stand in both while `signal-mark.js` is still being
 * fetched and permanently on browsers with no WebGL. The header therefore
 * never renders without its mark, and never reserves empty space for one.
 */
export function FlatMark({ className = 'h-6 w-6 text-accent' }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden="true">
      <rect x="2" y="9" width="3.5" height="6" rx="1.75" opacity="0.45" />
      <rect x="7.5" y="5" width="3.5" height="14" rx="1.75" opacity="0.7" />
      <rect x="13" y="2" width="3.5" height="20" rx="1.75" />
      <rect x="18.5" y="7" width="3.5" height="10" rx="1.75" opacity="0.55" />
    </svg>
  );
}
