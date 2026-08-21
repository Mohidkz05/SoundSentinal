'use client';

import { useEffect, useState } from 'react';

/**
 * The three guards every canvas in this app runs behind.
 *
 * The 3D layer is ambient — it is never the only thing carrying meaning — so
 * all three of these are allowed to switch it off entirely without the screen
 * losing anything a user needed.
 */

/**
 * `prefers-reduced-motion`, as a hook.
 *
 * Starts `true` deliberately. The server can't know the preference, so assuming
 * "reduced" means the first paint is always still and motion is only ever added
 * once we've confirmed it's wanted — never started and then snatched away.
 */
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(true);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => setReduced(mq.matches);
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, []);

  return reduced;
}

/**
 * True only when the element is near the viewport *and* the tab is foregrounded.
 *
 * Both halves matter: a scrolled-past hero and a backgrounded tab are the two
 * ways this app would otherwise burn a GPU rendering something nobody can see.
 */
export function useInView(ref, rootMargin = '240px') {
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let intersecting = false;
    let foreground = document.visibilityState === 'visible';
    const update = () => setInView(intersecting && foreground);

    const io = new IntersectionObserver(
      ([entry]) => {
        intersecting = entry.isIntersecting;
        update();
      },
      { rootMargin }
    );
    io.observe(el);

    const onVisibility = () => {
      foreground = document.visibilityState === 'visible';
      update();
    };
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      io.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [ref, rootMargin]);

  return inView;
}

/**
 * Whether this browser can give us a WebGL context at all.
 *
 * Cached: probing costs a throwaway context, and the answer cannot change
 * within a page load. A `false` here is not an error state — the DOM fallback
 * under every canvas is the real interface.
 */
let support = null;
export function hasWebGL() {
  if (support !== null) return support;
  if (typeof window === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    support =
      !!(window.WebGL2RenderingContext && canvas.getContext('webgl2')) ||
      !!(window.WebGLRenderingContext && canvas.getContext('webgl'));
  } catch {
    support = false;
  }
  return support;
}
