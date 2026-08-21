'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage, advance } from './stage';
import { usePalette } from './use-palette';

/**
 * The backdrop that sits behind every route.
 *
 * One fixed canvas in the root layout rather than a decorative canvas per
 * section: a browser will only hand out so many WebGL contexts, and this keeps
 * the count on any page to two or three.
 *
 * Deliberately near the threshold of visibility. The job is to give the page a
 * sense of depth behind the panels — the reason the UI reads as an instrument
 * on a surface rather than a document on a background — not to be noticed. If
 * you catch yourself watching it, the opacity is too high.
 */

const COUNT = 620;
const SPREAD = new THREE.Vector3(20, 12, 14);

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uPixelRatio;

  attribute float aSeed;
  attribute float aScale;

  varying float vFade;

  void main() {
    vec3 p = position;

    /* Each point drifts on its own phase. No shared beat, so the field never
       pulses in unison — that would read as a loading state. */
    float phase = aSeed * 6.2831853;
    p.y += sin(uTime * 0.22 + phase) * 0.45;
    p.x += cos(uTime * 0.17 + phase * 1.3) * 0.35;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);

    /* Fade with depth, and again towards the edges of the field, so the
       boundaries of the point cloud are never visible as an edge. */
    float depthFade = smoothstep(-16.0, -3.0, mv.z);
    float edgeFade = 1.0 - smoothstep(0.35, 0.5, length(p.xy / vec2(20.0, 12.0)));
    vFade = depthFade * edgeFade;

    gl_PointSize = aScale * uPixelRatio * (7.0 / max(-mv.z, 0.001));
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3  uColour;
  uniform float uOpacity;

  varying float vFade;

  void main() {
    /* Round the square point sprite off, and soften its edge — hard-edged
       squares at this size look like dead pixels. */
    float d = length(gl_PointCoord - 0.5);
    float mask = 1.0 - smoothstep(0.35, 0.5, d);
    if (mask <= 0.001) discard;

    gl_FragColor = vec4(uColour, uOpacity * vFade * mask);
  }
`;

function Depth() {
  const palette = usePalette();
  const pointsRef = useRef(null);

  const geometry = useMemo(() => {
    const positions = new Float32Array(COUNT * 3);
    const seeds = new Float32Array(COUNT);
    const scales = new Float32Array(COUNT);

    for (let i = 0; i < COUNT; i++) {
      positions[i * 3 + 0] = (Math.random() - 0.5) * SPREAD.x;
      positions[i * 3 + 1] = (Math.random() - 0.5) * SPREAD.y;
      positions[i * 3 + 2] = -Math.random() * SPREAD.z;
      seeds[i] = Math.random();
      scales[i] = 1.1 + Math.random() * 2.2;
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
    g.setAttribute('aScale', new THREE.BufferAttribute(scales, 1));
    return g;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  /* Uniforms are reached through the material ref, never through the object
     this component built — see the long note in `intake-field.js`. */
  const materialRef = useRef(null);

  const initialUniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uPixelRatio: { value: 1 },
      uColour: { value: new THREE.Color('#056e6a') },
      uOpacity: { value: 0.5 },
    }),
    []
  );

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uColour.value.copy(palette.accent);
  }, [palette]);

  useFrame((state, delta) => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    advance(u, delta);
    u.uPixelRatio.value = state.viewport.dpr;

    /* Pointer parallax, heavily damped. The lag is the point — an immediate
       response would feel like the background was tracking you. */
    const camera = state.camera;
    camera.position.x += (state.pointer.x * 0.55 - camera.position.x) * 0.018;
    camera.position.y += (state.pointer.y * 0.35 - camera.position.y) * 0.018;
    camera.lookAt(0, 0, -6);
  });

  return (
    <points ref={pointsRef} geometry={geometry}>
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

export function AmbientDepth() {
  return (
    <Stage
      className="fixed inset-0 -z-10"
      camera={{ position: [0, 0, 2], fov: 55 }}
      dpr={[1, 1.5]}
    >
      <Depth />
    </Stage>
  );
}
