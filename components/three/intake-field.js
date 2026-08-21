'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage, advance } from './stage';
import { usePalette } from './use-palette';

/**
 * The intake field — the upload screen's instrument face.
 *
 * A spectrogram is a surface: frequency across, time back, energy up. This is
 * that surface, built from bars and stood up in perspective, so the thing you
 * drop a clip onto is a picture of the representation the clip is about to be
 * turned into. It replaced a flat row of bars because a row can only show the
 * envelope; a surface can show the envelope *and* its history, which is what
 * the model actually consumes.
 *
 * It has four states and moves continuously between them:
 *
 *   - **Idle.** Two detuned travelling waves. Low, slow, no hurry.
 *   - **Hover.** A gaussian swell follows the pointer across the surface. The
 *     field is behind a `pointer-events-none` canvas, so the DOM drop target
 *     above it does the listening and hands coordinates down as props — the
 *     geometry never intercepts a drag.
 *   - **Drag.** Energy lifts globally and the whole surface leans in. This is
 *     the drag feedback: the instrument reacting, rather than a dashed border
 *     changing colour.
 *   - **Decoded.** The front row becomes the clip's real envelope and the rows
 *     behind it hold progressively attenuated copies, so the surface reads as
 *     the clip receding into its own history.
 *
 * All of it stays ambient under the rules in DESIGN.md: `aria-hidden`, no
 * pointer events, nothing here that the DOM does not already say, and absent
 * entirely without WebGL.
 */

const COLS = 72;
const ROWS = 16;
const WIDTH = 13.0;
const DEPTH = 7.0;
const FILL = 0.52; // share of each column slot the bar occupies

const vertexShader = /* glsl */ `
  uniform sampler2D uPeaks;
  uniform float uHasPeaks;
  uniform float uReveal;    // 0..1 as a decoded clip takes over from the idle wave
  uniform float uTime;
  uniform float uEnergy;    // lifts while a file is held over the target
  uniform vec2  uPointer;   // pointer in field space, 0..1 across and back
  uniform float uFocus;     // 0..1 presence of the pointer swell

  attribute vec2 aCell;     // this bar's place on the surface, 0..1 each way

  varying float vAmp;
  varying float vDepth;     // 0 at the front of the surface, 1 at the back
  varying float vRise;      // 0 at the bar's foot, 1 at its tip

  void main() {
    float u = aCell.x;
    /* aCell.y runs 0..1 from the far row to the near one, because that is the
       order the rows are generated in. Everything downstream wants distance
       *into* the scene, so flip it once here rather than remembering to invert
       it at each use — getting this backwards is what made the nearest row the
       faintest and the most attenuated. */
    float v = 1.0 - aCell.y;

    /* Idle: two detuned travelling waves, so the surface never visibly
       repeats and no row is ever a copy of its neighbour. */
    float idle = sin(u * 8.5 - uTime * 0.85 + v * 2.6)
               * cos(v * 4.2 + uTime * 0.42 - u * 1.7);
    idle = 0.24 + 0.20 * abs(idle);

    /* Decoded: the front row is the clip, and each row behind it is the same
       envelope carrying less energy — the clip receding into its history. */
    float recorded = texture2D(uPeaks, vec2(u, 0.5)).r;
    recorded *= 1.0 - 0.62 * v;

    float base = mix(idle, recorded, uHasPeaks * uReveal);

    /* Hover: a gaussian swell centred on the pointer. Tighter across than
       back, because the surface is foreshortened — an isotropic falloff in
       field space arrives on screen as an ellipse squashed the wrong way. */
    vec2 d = (vec2(u, v) - uPointer) * vec2(1.6, 1.0);
    float swell = exp(-dot(d, d) * 17.0) * uFocus;

    float amp = base * (1.0 + 0.55 * uEnergy) + swell * 0.52;

    /* A floor, so a silent stretch still draws a baseline tick. A surface with
       holes in it looks broken; a flat run looks like silence, which it is. */
    amp = max(amp, 0.02);

    vec3 p;
    p.x = (u - 0.5) * ${WIDTH.toFixed(1)} + position.x;
    p.y = position.y * amp * 2.05;   // position.y arrives as 0 or 1
    /* v counts *into* the scene, and the camera looks down +z, so the near row
       needs the larger z. */
    p.z = (0.5 - v) * ${DEPTH.toFixed(1)};

    vAmp = amp;
    vDepth = v;
    vRise = position.y;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3  uQuiet;
  uniform vec3  uLoud;
  uniform float uOpacity;

  varying float vAmp;
  varying float vDepth;
  varying float vRise;

  void main() {
    float k = clamp(vAmp, 0.0, 1.0);
    /* The idle surface should sit almost entirely at the quiet end — the
       accent is what a loud part of a real clip earns, not the resting
       state. These stops put idle around a third of the way up the mix. */
    vec3 colour = mix(uQuiet, uLoud, smoothstep(0.18, 0.92, k));

    /* Three fades, each doing a different job.

       Depth puts the back of the surface behind the front, so the perspective
       is legible without drawing a grid.

       Energy sits the quiet bars back, so the loud part of the clip is what the
       eye lands on.

       Rise is the important one. Every bar starts at the same floor, so near
       that floor a thousand translucent quads overlap and the bottom of the
       field turns into fog — which is exactly where the headline sits. Fading
       each bar out towards its own foot removes the accumulation at the source
       and leaves the bars looking like they are emerging from the surface
       rather than standing on it. */
    float depth = 1.0 - 0.62 * vDepth;
    float energy = 0.42 + 0.58 * smoothstep(0.0, 0.45, k);
    float rise = 0.22 + 0.78 * smoothstep(0.0, 0.85, vRise);

    gl_FragColor = vec4(colour, uOpacity * depth * energy * rise);
  }
`;

/** Pack a 0–1 envelope into a 1px-tall RGBA texture the shader can sample. */
function peaksTexture(peaks) {
  const width = peaks.length;
  const data = new Uint8Array(width * 4);
  for (let i = 0; i < width; i++) {
    const value = Math.round(Math.min(Math.max(peaks[i], 0), 1) * 255);
    data[i * 4 + 0] = value;
    data[i * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/* A single opaque-black pixel, so the sampler always has something bound and an
   unset envelope reads as zero rather than as noise. */
function emptyTexture() {
  const texture = new THREE.DataTexture(
    new Uint8Array([0, 0, 0, 255]),
    1,
    1,
    THREE.RGBAFormat
  );
  texture.needsUpdate = true;
  return texture;
}

function Surface({ peaks, energy, pointerRef }) {
  const palette = usePalette();
  const groupRef = useRef(null);
  const blank = useMemo(emptyTexture, []);

  /* Keep the surface inside the frame whatever the container's aspect ratio
     is. A field that runs off the edge has lost the property that makes it
     worth showing — that it is one whole surface. */
  const viewport = useThree((state) => state.viewport);
  const fit = Math.min(1, (viewport.width * 0.98) / WIDTH);

  const geometry = useMemo(() => {
    const slot = WIDTH / COLS;
    const bar = slot * FILL;
    const quads = COLS * ROWS;

    const positions = new Float32Array(quads * 4 * 3);
    const cells = new Float32Array(quads * 4 * 2);
    const indices = new Uint32Array(quads * 6);

    let q = 0;
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const u = (col + 0.5) / COLS;
        const v = ROWS === 1 ? 0 : row / (ROWS - 1);

        /* x is carried in `position` as an offset from the column's centre;
           the shader places the column itself from `aCell.x`. Keeping the two
           separate is what lets the bar keep its width while the surface
           stretches. */
        const half = bar / 2;
        const corners = [
          [-half, 0],
          [half, 0],
          [half, 1],
          [-half, 1],
        ];

        const base = q * 4;
        corners.forEach(([x, y], c) => {
          positions[(base + c) * 3 + 0] = x;
          positions[(base + c) * 3 + 1] = y;
          positions[(base + c) * 3 + 2] = 0;
          cells[(base + c) * 2 + 0] = u;
          cells[(base + c) * 2 + 1] = v;
        });

        const t = q * 6;
        indices[t + 0] = base;
        indices[t + 1] = base + 1;
        indices[t + 2] = base + 2;
        indices[t + 3] = base;
        indices[t + 4] = base + 2;
        indices[t + 5] = base + 3;

        q += 1;
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('aCell', new THREE.BufferAttribute(cells, 2));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    return g;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  /* ------------------------------------------------------------------------
     Uniforms are reached through a ref on the material, never through a
     captured object.

     The obvious version of this — build a `uniforms` object with `useMemo`,
     pass it as a JSX prop, and mutate `uniforms.uTime.value` in `useFrame` —
     does not work, and fails silently. The object the component holds is not
     the object the rendered material ends up using, so every per-frame write
     lands somewhere the GPU never reads. Nothing errors; the scene simply
     renders with whatever the uniforms were at creation, which looks like a
     tuning problem rather than a wiring one. It cost a long time to find:
     setting `uOpacity` to 0 every frame left the surface fully visible, which
     is what finally proved the writes were going nowhere.

     Reading `materialRef.current.uniforms` means we mutate whatever object the
     renderer is actually using, whatever R3F did with the prop.
     --------------------------------------------------------------------- */
  const materialRef = useRef(null);

  const initialUniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uPeaks: { value: blank },
      uHasPeaks: { value: 0 },
      uReveal: { value: 0 },
      uEnergy: { value: 0 },
      uPointer: { value: new THREE.Vector2(0.5, 0.5) },
      uFocus: { value: 0 },
      uQuiet: { value: new THREE.Color('#bfcac9') },
      uLoud: { value: new THREE.Color('#056e6a') },
      uOpacity: { value: 0.92 },
    }),
    [blank]
  );

  /* `muted` rather than `lineStrong` for the quiet end. A hairline grey is
     the right weight for a 1px rule but disappears entirely as a bar on the
     light canvas, which left the low parts of the surface invisible and the
     field reading as a few floating peaks. */
  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uQuiet.value.copy(palette.muted);
    u.uLoud.value.copy(palette.accent);
  }, [palette]);

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return undefined;
    if (!peaks) {
      u.uHasPeaks.value = 0;
      return undefined;
    }
    const texture = peaksTexture(peaks);
    u.uPeaks.value = texture;
    u.uHasPeaks.value = 1;
    // Re-run the reveal so a second file animates in like the first.
    u.uReveal.value = 0;
    return () => {
      texture.dispose();
      u.uPeaks.value = blank;
    };
  }, [peaks, blank]);

  useEffect(() => () => blank.dispose(), [blank]);

  useFrame((_, delta) => {
    const u = materialRef.current?.uniforms;
    if (!u) return;

    advance(u, delta);

    /* Everything eases rather than snaps. The pointer swell in particular has
       to lag the cursor slightly or it reads as a hard spotlight stuck to the
       mouse instead of a surface responding to it. */
    const step = Math.min(1, delta * 7.5);

    u.uReveal.value += ((peaks ? 1 : 0) - u.uReveal.value) * 0.05;
    u.uEnergy.value += (energy - u.uEnergy.value) * step;

    /* The pointer is read out of a ref rather than taken as a prop. A prop would
       mean a React re-render on every mousemove — sixty times a second, for a
       value only the render loop consumes — and the loop is already running, so
       it can simply look at the latest value when it needs it. */
    const p = pointerRef?.current;
    u.uFocus.value += ((p?.active ? 1 : 0) - u.uFocus.value) * step;
    if (p?.active) {
      /* The pointer arrives in screen coordinates, where y grows downwards, and
         the near edge of the surface is at the *bottom* of the panel. The shader
         measures depth into the scene, so y has to be flipped to land the swell
         under the cursor instead of mirrored across the surface. */
      u.uPointer.value.x += (p.x - u.uPointer.value.x) * step;
      u.uPointer.value.y += (1 - p.y - u.uPointer.value.y) * step;
    }

    if (groupRef.current) {
      const t = u.uTime.value;
      /* A slow drift, plus a lean towards the pointer. The lean is small on
         purpose: the surface should acknowledge the cursor, not chase it. */
      const leanX = p?.active ? (p.x - 0.5) * 0.16 : 0;
      const leanY = p?.active ? (p.y - 0.5) * 0.10 : 0;

      groupRef.current.rotation.y +=
        (Math.sin(t * 0.16) * 0.07 + leanX - groupRef.current.rotation.y) * 0.06;
      groupRef.current.rotation.x +=
        (-0.54 + Math.sin(t * 0.11) * 0.02 - leanY - groupRef.current.rotation.x) * 0.06;
    }
  });

  return (
    /* Sunk below the camera's centre line so the surface reads as ground the
       page stands on, and the copy at the bottom of the panel gets a horizon
       above it rather than a wall behind it. */
    <group ref={groupRef} position={[0, -2.30, 0]}>
      <mesh geometry={geometry} scale={[fit, 1, 1]}>
        <shaderMaterial
          ref={materialRef}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          uniforms={initialUniforms}
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

/**
 * @param {Float32Array|null} peaks   normalised envelope, or null to idle
 * @param {boolean}          active   a file is being dragged over the target
 * @param {object}      pointerRef    a ref holding `{ x, y, active }` in field
 *                                    space (0–1 each way, y measured from the
 *                                    top). Written by the DOM drop target, read
 *                                    by the render loop — never a prop, so the
 *                                    page does not re-render per mousemove.
 */
export function IntakeField({
  peaks = null,
  active = false,
  pointerRef = null,
  className = '',
}) {
  return (
    <Stage className={className} camera={{ position: [0, 3.9, 8.0], fov: 40 }}>
      <Surface peaks={peaks} energy={active ? 1 : 0} pointerRef={pointerRef} />
    </Stage>
  );
}
