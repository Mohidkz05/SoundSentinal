import { clsx } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

/* tailwind-merge must know the design system's own type scale (globals.css,
   `--text-*`). Unknown, `text-body` and `text-secondary` both look like a
   text colour, and merging them would silently drop one. */
const merge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [{ text: ['display', 'h1', 'h2', 'h3', 'lead', 'body', 'small', 'label', 'readout', 'score', 'tick'] }],
    },
  },
});

/** shadcn/ui's class combiner: conditional classes, later ones win. */
export function cn(...inputs) {
  return merge(clsx(inputs));
}
