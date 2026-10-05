'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import {
  hasWebGL,
  stepDownQuality,
  useInView,
  useMotionPaused,
  usePrefersReducedMotion,
  useQualityTier,
} from './use-stage';

/**
 * The one way a canvas enters this app.
 *
 * Every piece of 3D in SoundSentinal mounts through here so the guarantees are
 * made once rather than remembered five times:
 *
 *   - **Nothing renders on the server.** WebGL has no SSR story; the canvas
 *     appears after mount, under whatever DOM the caller passed as children.
 *   - **Offscreen and backgrounded canvases stop.** `frameloop` drops to
 *     `never`, which halts the render loop rather than merely hiding it.
 *   - **Reduced motion means one frame, not no frame.** `demand` renders the
 *     scene once and then sits still, so the composition survives while the
 *     movement doesn't. Removing it entirely would punish the preference.
 *   - **No WebGL means no canvas.** The DOM underneath is the real interface;
 *     this layer is always additive.
 *   - **Decoration is invisible to assistive tech** and never eats a pointer
 *     event.
 *
 *   - **The visitor can stop it.** The header's pause switch holds every
 *     canvas still (WCAG 2.2.2), the same way reduced motion does.
 *   - **A slow page sheds the layer, not the interface.** The frame-rate
 *     governor below steps the page-wide quality tier down: first to 1×
 *     pixels, then to a still frame.
 *
 * `dpr` is capped below the device ratio on purpose. These are soft, low-
 * contrast fields where the extra samples of a 3× retina buffer cost fill rate
 * and buy nothing you can see.
 */
export function Stage({
  children,
  className = '',
  style,
  camera = { position: [0, 0, 5], fov: 45 },
  dpr = [1, 1.75],
  /** Off for full-viewport soft fields, where MSAA is pure fill cost. */
  antialias = true,
  /** Rendered instead of the canvas when WebGL is unavailable. */
  fallback = null,
  ...canvasProps
}) {
  const hostRef = useRef(null);
  const inView = useInView(hostRef);
  const reduced = usePrefersReducedMotion();
  const tier = useQualityTier();
  const paused = useMotionPaused();

  // Deferred to an effect so the server and the first client render agree.
  const [enabled, setEnabled] = useState(false);
  useEffect(() => setEnabled(hasWebGL()), []);

  const still = reduced || paused || tier >= 2;
  const frameloop = still ? 'demand' : inView ? 'always' : 'never';

  return (
    <div
      ref={hostRef}
      aria-hidden="true"
      className={`pointer-events-none select-none ${className}`}
      style={style}
    >
      {enabled ? (
        <Canvas
          frameloop={frameloop}
          dpr={tier >= 1 ? 1 : dpr}
          camera={camera}
          gl={{ antialias, alpha: true, powerPreference: 'low-power' }}
          {...canvasProps}
        >
          {!still && <Governor key={tier} tier={tier} />}
          {children}
        </Canvas>
      ) : (
        fallback
      )}
    </div>
  );
}

/**
 * Watches the frame rate and steps the page's quality tier down when it is too
 * low to be worth the cost. Two seconds under 40 fps per step, after a one-
 * second grace period for shader compilation. Gaps over 250 ms are a paused
 * loop (offscreen, backgrounded tab), not a slow frame, and are skipped.
 */
const GRACE_S = 1;
const WINDOW_S = 2;
const MIN_FPS = 40;

function Governor({ tier }) {
  const acc = useRef({ age: 0, time: 0, frames: 0 });
  useFrame((_, delta) => {
    const a = acc.current;
    if (delta > 0.25) return;
    a.age += delta;
    if (a.age < GRACE_S) return;
    a.time += delta;
    a.frames += 1;
    if (a.time < WINDOW_S) return;
    const fps = a.frames / a.time;
    a.time = 0;
    a.frames = 0;
    // Keyed on the tier, so the next tier starts over with its own grace.
    if (fps < MIN_FPS) stepDownQuality(tier);
  });
  return null;
}

/**
 * Shared animation clock.
 *
 * Scenes advance their own `uTime` from this rather than from
 * `state.clock.elapsedTime`, because a canvas that was paused offscreen would
 * otherwise resume with a large time jump and visibly snap. Accumulating delta
 * only while running means a field picks up exactly where it left off.
 */
export function advance(uniforms, delta, scale = 1) {
  uniforms.uTime.value += Math.min(delta, 0.05) * scale;
}
