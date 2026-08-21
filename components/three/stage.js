'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { hasWebGL, useInView, usePrefersReducedMotion } from './use-stage';

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
  /** Rendered instead of the canvas when WebGL is unavailable. */
  fallback = null,
  ...canvasProps
}) {
  const hostRef = useRef(null);
  const inView = useInView(hostRef);
  const reduced = usePrefersReducedMotion();

  // Deferred to an effect so the server and the first client render agree.
  const [enabled, setEnabled] = useState(false);
  useEffect(() => setEnabled(hasWebGL()), []);

  const frameloop = reduced ? 'demand' : inView ? 'always' : 'never';

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
          dpr={dpr}
          camera={camera}
          gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
          {...canvasProps}
        >
          {children}
        </Canvas>
      ) : (
        fallback
      )}
    </div>
  );
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
