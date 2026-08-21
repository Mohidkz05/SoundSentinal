'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage, advance } from './stage';
import { usePalette } from './use-palette';

/**
 * The upload screen's display — your clip, not a stock animation.
 *
 * With no file it idles on a procedural envelope. Once a clip is decoded the
 * envelope is uploaded as a texture and the bars become the actual shape of
 * that recording, which does a job no placeholder can: it proves the browser
 * read the file, and it lets you see at a glance that you picked the right one.
 *
 * Bars rather than a filled ribbon. A translucent band with a soft centre
 * reads as a smudge at this size, where a bar display reads as audio
 * immediately — and it is the same figure as the wordmark, which is a
 * spectrogram slice, so the two belong to one another.
 *
 * The envelope is sampled in the vertex shader from a 1-pixel-tall data
 * texture, so re-shaping the display costs one texture upload rather than
 * rebuilding geometry.
 */

const BARS = 96;
const WIDTH = 10.5;
const FILL = 0.55; // share of each slot the bar occupies; the rest is the gap

const vertexShader = /* glsl */ `
  uniform sampler2D uPeaks;
  uniform float uHasPeaks;
  uniform float uTime;
  uniform float uReveal;   // 0..1 as a decoded clip takes over from the idle wave
  uniform float uEnergy;   // lifts while a file is being dragged over the target
  uniform float uHeight;

  attribute float aU;      // this bar's sampling point along the clip, 0..1

  varying float vAmp;

  void main() {
    /* The idle wave: two detuned sines, so it never visibly repeats. */
    float idle = sin(aU * 17.0 + uTime * 0.9) * sin(aU * 6.3 - uTime * 0.55);
    idle = (0.40 + 0.22 * uEnergy) * abs(idle) + 0.06;

    float recorded = texture2D(uPeaks, vec2(aU, 0.5)).r;

    float amp = mix(idle, recorded, uHasPeaks * uReveal);
    /* A floor, so silence still draws a baseline tick. A display with gaps in
       it looks broken; a flat run looks like silence, which is what it is. */
    amp = max(amp, 0.022);

    vec3 p = position;
    p.y = position.y * amp * uHeight;   // position.y arrives as -1 or +1

    /* A slow travelling wave through depth, constant across each bar so the
       bars stay rectangular. The envelope is accurate in y; z only ever
       carries the presentation. */
    p.z += sin(aU * 3.4 - uTime * 0.5) * 0.30;

    vAmp = amp;
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
    vec3 colour = mix(uQuiet, uLoud, smoothstep(0.02, 0.42, k));
    /* Quiet bars sit back rather than drawing at full strength, so the loud
       part of the clip is what the eye lands on. */
    float alpha = uOpacity * (0.42 + 0.58 * smoothstep(0.0, 0.35, k));
    gl_FragColor = vec4(colour, alpha);
  }
`;

/** Pack a 0–1 envelope into a 1px-tall RGBA texture the shader can sample. */
function peaksTexture(peaks) {
  const width = peaks.length;
  const data = new Uint8Array(width * 4);
  for (let i = 0; i < width; i++) {
    const v = Math.round(Math.min(Math.max(peaks[i], 0), 1) * 255);
    data[i * 4 + 0] = v;
    data[i * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/* A single opaque-black pixel, so the sampler always has something bound and
   an unset envelope reads as zero rather than as noise. */
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

function Bars({ peaks, energy }) {
  const palette = usePalette();
  const meshRef = useRef(null);
  const blank = useMemo(emptyTexture, []);

  /* Keep the whole display inside the frame whatever the container's aspect
     ratio is. A waveform that runs off the edge has lost the one property
     that makes it worth showing — that it is the whole clip. */
  const viewport = useThree((state) => state.viewport);
  const fit = Math.min(1, (viewport.width * 0.94) / WIDTH);

  const geometry = useMemo(() => {
    const slot = WIDTH / BARS;
    const bar = slot * FILL;

    const positions = new Float32Array(BARS * 4 * 3);
    const us = new Float32Array(BARS * 4);
    const indices = new Uint16Array(BARS * 6);

    for (let i = 0; i < BARS; i++) {
      const x0 = -WIDTH / 2 + slot * i + (slot - bar) / 2;
      const x1 = x0 + bar;
      const u = (i + 0.5) / BARS;
      const v = i * 4;

      // Bottom-left, bottom-right, top-right, top-left.
      const corners = [
        [x0, -1],
        [x1, -1],
        [x1, 1],
        [x0, 1],
      ];
      corners.forEach(([x, y], c) => {
        positions[(v + c) * 3 + 0] = x;
        positions[(v + c) * 3 + 1] = y;
        positions[(v + c) * 3 + 2] = 0;
        us[v + c] = u;
      });

      const t = i * 6;
      indices[t + 0] = v;
      indices[t + 1] = v + 1;
      indices[t + 2] = v + 2;
      indices[t + 3] = v;
      indices[t + 4] = v + 2;
      indices[t + 5] = v + 3;
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('aU', new THREE.BufferAttribute(us, 1));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    return g;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  /* Uniforms are reached through the material ref, never through the object
     this component built. See the long note in `intake-field.js`: mutating the
     memoised object is a silent no-op, because it is not the object the
     rendered material uses, and every animated uniform in this file was going
     nowhere as a result. */
  const materialRef = useRef(null);

  const initialUniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uPeaks: { value: blank },
      uHasPeaks: { value: 0 },
      uReveal: { value: 0 },
      uEnergy: { value: 0 },
      uHeight: { value: 1.05 },
      uQuiet: { value: new THREE.Color('#bfcac9') },
      uLoud: { value: new THREE.Color('#056e6a') },
      uOpacity: { value: 0.96 },
    }),
    [blank]
  );

  useEffect(() => {
    const u = materialRef.current?.uniforms;
    if (!u) return;
    u.uQuiet.value.copy(palette.lineStrong);
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

    const target = peaks ? 1 : 0;
    u.uReveal.value += (target - u.uReveal.value) * 0.05;
    u.uEnergy.value += (energy - u.uEnergy.value) * 0.08;

    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(u.uTime.value * 0.19) * 0.13;
      meshRef.current.rotation.x = -0.17 + Math.sin(u.uTime.value * 0.13) * 0.035;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} scale={[fit, 1, 1]}>
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
  );
}

/**
 * @param {Float32Array|null} peaks  normalised envelope, or null to idle
 * @param {boolean} active           a file is being dragged over the target
 */
export function WaveformDisplay({ peaks = null, active = false, className = '' }) {
  return (
    <Stage className={className} camera={{ position: [0, 0, 3.7], fov: 42 }}>
      <Bars peaks={peaks} energy={active ? 1 : 0} />
    </Stage>
  );
}
