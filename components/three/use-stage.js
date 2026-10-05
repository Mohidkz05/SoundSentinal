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
 * Whether this browser can give us a *hardware* WebGL context.
 *
 * A software rasteriser counts as no. With hardware acceleration off (or the
 * GPU blocklisted), browsers fall back to SwiftShader, llvmpipe or Windows'
 * Basic Render Driver, which run these full-viewport shaders on the CPU — the
 * page stutters to buy decoration. Set localStorage['soundsentinal-3d'] =
 * 'force' to render anyway; the headless test harness needs that, because
 * SwiftShader is its only WebGL.
 *
 * Cached: probing costs a throwaway context (released here), and the answer
 * cannot change within a page load. A `false` here is not an error state — the
 * DOM fallback under every canvas is the real interface.
 */
const SOFTWARE_RENDERER = /swiftshader|llvmpipe|softpipe|basic render|software/i;

let support = null;
export function hasWebGL() {
  if (support !== null) return support;
  if (typeof window === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
      support = false;
    } else {
      const info = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = String(
        gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER)
      );
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      support = !SOFTWARE_RENDERER.test(renderer) || forced();
    }
  } catch {
    support = false;
  }
  return support;
}

function forced() {
  try {
    return window.localStorage.getItem('soundsentinal-3d') === 'force';
  } catch {
    return false;
  }
}

/**
 * Page-wide quality tier: 0 full, 1 device-pixel ratio pinned to 1, 2 frozen
 * (every canvas holds its last frame, as under reduced motion).
 *
 * Shared rather than per canvas because frame rate is a property of the page:
 * all canvases draw in the same animation frame, so if one is slow they all
 * are, and a canvas mounted later should start at the tier already found.
 * Only ever steps down within a page load.
 */
let tier = 0;
const tierListeners = new Set();

export function useQualityTier() {
  const [value, setValue] = useState(tier);
  useEffect(() => {
    tierListeners.add(setValue);
    setValue(tier);
    return () => tierListeners.delete(setValue);
  }, []);
  return value;
}

/** `from` is the tier the caller measured at: every canvas runs a governor,
 *  and one slow window must cost one step, not one per canvas. */
export function stepDownQuality(from) {
  if (from !== tier || tier >= 2) return;
  tier += 1;
  tierListeners.forEach((set) => set(tier));
}

/**
 * The visitor's own "pause motion" switch, in the header.
 *
 * WCAG 2.2.2: motion that starts by itself, runs longer than five seconds and
 * sits beside other content needs a way to stop it. prefers-reduced-motion
 * covers people who set it at the OS level; this covers everyone else, on any
 * page. Paused, every canvas holds its current frame — the same still state
 * reduced motion uses. Remembered across visits.
 */
const MOTION_KEY = 'soundsentinal-motion';
let paused = null;
const pausedListeners = new Set();

function readPaused() {
  try {
    return window.localStorage.getItem(MOTION_KEY) === 'paused';
  } catch {
    return false;
  }
}

export function useMotionPaused() {
  // false on the server and the first client render, so they agree.
  const [value, setValue] = useState(false);
  useEffect(() => {
    if (paused === null) paused = readPaused();
    setValue(paused);
    pausedListeners.add(setValue);
    return () => pausedListeners.delete(setValue);
  }, []);
  return value;
}

export function setMotionPaused(next) {
  paused = next;
  try {
    window.localStorage.setItem(MOTION_KEY, next ? 'paused' : 'playing');
  } catch {
    // Storage blocked: the switch still works for this page load.
  }
  pausedListeners.forEach((set) => set(next));
}
