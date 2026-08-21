'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage, advance } from './stage';
import { usePalette, tierColor } from './use-palette';
import { usePrefersReducedMotion } from './use-stage';
import { tierFor } from '../../src/lib/verdict';

/**
 * The result screen's field: order as certainty.
 *
 * A lattice of points that holds its grid when the reading is far from the
 * decision threshold and scatters as it approaches — so the interface visibly
 * loses its composure exactly where the model does. It is the third rule in
 * DESIGN.md ("the colour drains when we are unsure") stated a second way, in
 * arrangement rather than chroma.
 *
 * It carries no number and no verdict. The calibration meter is still the only
 * thing on the page reporting a result; if this fails to load nothing legible
 * is lost. It takes its colour from the verdict scale rather than the brand
 * teal, because anything coloured in service of a result must come from the
 * scale.
 *
 * It is deliberately mounted *around* the meter panel rather than inside it.
 * The panel is dense with small type at every height, so a field behind it
 * would sit under a label wherever it was placed; the opaque panel masks the
 * middle of this one and it is only ever seen in the margins — the room around
 * the instrument, not a texture over the readout.
 */

const COLUMNS = 52;
const ROWS = 17;
const FIELD_WIDTH = 11.2;
const FIELD_HEIGHT = 3.6;

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uScatter;     // 0 = ordered lattice, 1 = fully unsettled
  uniform float uPixelRatio;

  attribute vec3  aDrift;
  attribute float aSeed;

  varying float vAlpha;

  void main() {
    vec3 p = position;

    float phase = aSeed * 6.2831853;
    /* Each point wanders along its own fixed vector. Scaling that vector by
       uScatter means the lattice dissolves continuously rather than switching
       between two states. */
    p += aDrift * uScatter * (0.65 + 0.35 * sin(uTime * 0.6 + phase));

    /* A small always-on breath, so a confident reading still looks alive
       rather than frozen. */
    p.z += sin(uTime * 0.35 + phase) * 0.08;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);

    /* Fade towards the boundary of the lattice in both axes, so the field has
       no visible edge and simply thins out into the page. */
    vec2 q = p.xy / vec2(11.2, 3.6);
    vAlpha = 1.0 - smoothstep(0.26, 0.5, length(q));

    gl_PointSize = uPixelRatio * (2.3 + 1.6 * uScatter) * (4.6 / max(-mv.z, 0.001));
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3  uColour;
  uniform float uOpacity;

  varying float vAlpha;

  void main() {
    float d = length(gl_PointCoord - 0.5);
    float mask = 1.0 - smoothstep(0.32, 0.5, d);
    if (mask <= 0.001) discard;

    gl_FragColor = vec4(uColour, uOpacity * vAlpha * mask);
  }
`;

/**
 * How unsettled the field should be: 1 when the reading sits on the threshold,
 * falling to 0 as it moves half the scale away. This is the same quantity the
 * verdict scale expresses as loss of chroma.
 */
function scatterFor(probability, threshold) {
  const distance = Math.abs(probability - threshold);
  return 1 - Math.min(1, distance / 0.5);
}

function Field({ probability, threshold }) {
  const palette = usePalette();
  const reduced = usePrefersReducedMotion();
  const invalidate = useThree((state) => state.invalidate);
  const target = scatterFor(probability, threshold);

  const geometry = useMemo(() => {
    const count = COLUMNS * ROWS;
    const positions = new Float32Array(count * 3);
    const drift = new Float32Array(count * 3);
    const seeds = new Float32Array(count);

    let i = 0;
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        positions[i * 3 + 0] = -FIELD_WIDTH / 2 + (FIELD_WIDTH * col) / (COLUMNS - 1);
        positions[i * 3 + 1] = -FIELD_HEIGHT / 2 + (FIELD_HEIGHT * row) / (ROWS - 1);
        positions[i * 3 + 2] = 0;

        /* Scaled against the lattice spacing (~0.22 here), not chosen for
           looks. Below roughly one cell the grid still reads as a grid; well
           above it the order is gone. That relationship is the whole signal —
           if the drift were large the field would look random at every
           reading and say nothing. */
        drift[i * 3 + 0] = (Math.random() - 0.5) * 0.52;
        drift[i * 3 + 1] = (Math.random() - 0.5) * 0.42;
        drift[i * 3 + 2] = (Math.random() - 0.5) * 0.6;

        seeds[i] = Math.random();
        i++;
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('aDrift', new THREE.BufferAttribute(drift, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
    return g;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  /* Uniforms are reached through the material ref, never through the object
     this component built — see the long note in `intake-field.js`. */
  const materialRef = useRef(null);

  const initialUniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uScatter: { value: target },
      uPixelRatio: { value: 1 },
      uColour: { value: new THREE.Color('#6a549a') },
      uOpacity: { value: 0.62 },
    }),
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uColour.value.copy(tierColor(palette, tierFor(probability).id));
    invalidate();
  }, [palette, probability, invalidate]);

  /* Under reduced motion the loop only runs on demand, so the eased approach
     below never converges. Snap instead, and ask for the one frame. */
  useEffect(() => {
    if (!reduced) return;
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uScatter.value = target;
    invalidate();
  }, [reduced, target, invalidate]);

  useFrame((state, delta) => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    advance(u, delta);
    u.uPixelRatio.value = state.viewport.dpr;
    u.uScatter.value += (target - u.uScatter.value) * 0.045;
  });

  return (
    <points geometry={geometry}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={initialUniforms}
        transparent
        depthWrite={false}
      />
    </points>
  );
}

export function UncertaintyField({ probability = 0, threshold = 0.5, className = '' }) {
  return (
    <Stage className={className} camera={{ position: [0, 0, 4.4], fov: 46 }}>
      <Field probability={probability} threshold={threshold} />
    </Stage>
  );
}
