'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage, advance } from './stage';
import { usePalette } from './use-palette';

/**
 * The hero field: a log-Mel spectrogram, drawn as a ridgeline in 3D.
 *
 * This is the one image in the product that shows what the model actually sees.
 * `preprocess_waveform()` turns every clip into a 128-bin log-Mel spectrogram
 * before the CNN ever touches it, so the hero is a picture of the input format
 * rather than an abstract "AI" motif — which is also why it can be brand teal
 * without breaking the rule that teal never renders a result. Nothing here is a
 * reading; it is the substrate a reading is taken from.
 *
 * Rows are frames, the x axis is mel bins, height is energy. The surface is
 * generated in the vertex shader from drifting formant bands, so the CPU does
 * no per-frame work and the whole field costs one draw call.
 */

/* Sized to the frame, not to taste. The camera below sees roughly 5.5 world
   units across, so a wider field would crop to a single formant and the
   structure that makes this legible as a spectrogram — separated bands at
   fixed mel positions — would be off screen. */
const ROWS = 46;
const COLS = 150;
const WIDTH = 5.6;
const DEPTH = 4.2;

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform vec2  uSize;
  uniform float uLift;

  varying float vAmp;

  /* A Gaussian bump: one formant. Cheaper than noise and, unlike noise, it
     produces the horizontally banded structure voiced speech actually has. */
  float band(float x, float centre, float width) {
    float d = (x - centre) / width;
    return exp(-d * d);
  }

  void main() {
    vec3 p = position;

    float u = p.x / uSize.x + 0.5;   // mel bin,  0..1
    float v = p.z / uSize.y + 0.5;   // frame,    0..1
    float t = uTime + v * 1.6;       // later frames lag, so the field scrolls

    float a = 0.0;
    a += 0.90 * band(u, 0.15 + 0.030 * sin(t * 1.70), 0.050);
    a += 0.58 * band(u, 0.33 + 0.045 * sin(t * 1.10 + 1.7), 0.070);
    a += 0.34 * band(u, 0.57 + 0.055 * sin(t * 0.80 + 3.1), 0.095);
    /* High mel bins: low-level broadband texture. This is the region the Mel
       scale compresses and where vocoder artefacts live — see the LFCC note in
       CLAUDE.md. */
    a += 0.10 * band(u, 0.84, 0.170) * (0.5 + 0.5 * sin(t * 3.3 + u * 40.0));

    /* Breathe, so it reads as speech rather than a constant tone. The floor is
       high enough that the field never fully flattens — a field that empties
       out looks like it has stalled. */
    a *= 0.52 + 0.48 * pow(0.5 + 0.5 * sin(t * 0.85), 1.5);

    /* Fade the near and far edges to nothing so the field has no hard border
       to give away that it is a finite plane. */
    a *= smoothstep(0.0, 0.14, v) * (1.0 - smoothstep(0.84, 1.0, v));

    p.y += a * uLift;
    vAmp = a;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3  uQuiet;
  uniform vec3  uLoud;
  uniform float uOpacity;

  varying float vAmp;

  void main() {
    float k = clamp(vAmp, 0.0, 1.0);
    /* Reaches full brand teal around two-thirds amplitude rather than at the
       very peak, so the ridges read as coloured signal instead of a grey mesh
       with a few bright crests. */
    vec3 colour = mix(uQuiet, uLoud, smoothstep(0.03, 0.48, k));

    /* Quiet bins fade out rather than drawing a full-strength grid. Without
       this the field reads as a wireframe object; with it, it reads as signal. */
    float alpha = uOpacity * (0.07 + 0.93 * smoothstep(0.0, 0.5, k));

    gl_FragColor = vec4(colour, alpha);
  }
`;

function Field({ opacity }) {
  const palette = usePalette();
  const groupRef = useRef(null);
  const { camera } = useThree();

  useEffect(() => {
    camera.lookAt(0, 0.15, 0);
  }, [camera]);

  const geometry = useMemo(() => {
    const positions = new Float32Array(ROWS * (COLS - 1) * 2 * 3);
    let i = 0;
    for (let r = 0; r < ROWS; r++) {
      const z = -DEPTH / 2 + (DEPTH * r) / (ROWS - 1);
      for (let c = 0; c < COLS - 1; c++) {
        const x0 = -WIDTH / 2 + (WIDTH * c) / (COLS - 1);
        const x1 = -WIDTH / 2 + (WIDTH * (c + 1)) / (COLS - 1);
        positions[i++] = x0; positions[i++] = 0; positions[i++] = z;
        positions[i++] = x1; positions[i++] = 0; positions[i++] = z;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return g;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  /* Uniforms are reached through the material ref, never through the object
     this component built — see the long note in `intake-field.js`. */
  const materialRef = useRef(null);

  const initialUniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uSize: { value: new THREE.Vector2(WIDTH, DEPTH) },
      uLift: { value: 0.78 },
      uQuiet: { value: new THREE.Color('#7d8988') },
      uLoud: { value: new THREE.Color('#056e6a') },
      uOpacity: { value: opacity },
    }),
    // Built once; values are pushed in below so a theme change never remounts
    // the material.
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uQuiet.value.copy(palette.line);
    u.uLoud.value.copy(palette.accent);
  }, [palette]);

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uOpacity.value = opacity;
  }, [opacity]);

  useFrame((_, delta) => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    advance(u, delta, 0.16);
    if (groupRef.current) {
      // A slow yaw, ~40s per cycle. Enough to keep the perspective alive,
      // far too slow to pull the eye off the headline beside it.
      groupRef.current.rotation.y = Math.sin(u.uTime.value * 0.09) * 0.05;
    }
  });

  return (
    <group ref={groupRef} rotation={[0, 0, 0]}>
      <lineSegments geometry={geometry}>
        <shaderMaterial
          ref={materialRef}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          uniforms={initialUniforms}
          transparent
          depthWrite={false}
        />
      </lineSegments>
    </group>
  );
}

export function SpectralField({ className = '', style, opacity = 0.9 }) {
  return (
    <Stage
      className={className}
      style={style}
      camera={{ position: [0, 1.15, 3.9], fov: 42 }}
    >
      <Field opacity={opacity} />
    </Stage>
  );
}
