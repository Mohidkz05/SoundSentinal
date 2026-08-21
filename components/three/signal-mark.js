'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Stage } from './stage';
import { usePalette } from './use-palette';
import { FlatMark } from '../ui/mark';

/**
 * The wordmark's four bars, rebuilt as geometry.
 *
 * Same silhouette as the flat mark it replaces — a spectrogram slice — so the
 * brand doesn't change, only its depth. The bars settle on a slow level-meter
 * motion rather than spinning: this element is in the sticky header on every
 * page, and anything with a noticeable beat up there would be exhausting by the
 * second screen.
 *
 * Falls back to the original SVG when WebGL is unavailable, so the header is
 * never missing its mark.
 */

/* Height and phase per bar, matching the proportions of the SVG mark. */
const BARS = [
  { x: -0.63, height: 0.34, phase: 0.0 },
  { x: -0.21, height: 0.74, phase: 1.9 },
  { x: 0.21, height: 1.0, phase: 3.4 },
  { x: 0.63, height: 0.52, phase: 5.1 },
];

const BAR_WIDTH = 0.3;

function Bars() {
  const palette = usePalette();
  const groupRef = useRef(null);
  const barRefs = useRef([]);
  const timeRef = useRef(0);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#056e6a'),
        roughness: 0.42,
        metalness: 0.08,
      }),
    []
  );

  useEffect(() => {
    material.color.copy(palette.accent);
  }, [palette, material]);

  useEffect(() => () => material.dispose(), [material]);

  useFrame((_, delta) => {
    timeRef.current += Math.min(delta, 0.05);
    const t = timeRef.current;

    BARS.forEach((bar, i) => {
      const mesh = barRefs.current[i];
      if (!mesh) return;
      // ~7s period, ±14%. A level meter idling, not reacting.
      const k = 1 + Math.sin(t * 0.9 + bar.phase) * 0.14;
      mesh.scale.y = k;
    });

    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(t * 0.28) * 0.34;
    }
  });

  return (
    <>
      <ambientLight intensity={1.7} />
      <directionalLight position={[2, 3, 4]} intensity={2.1} />
      <directionalLight position={[-3, -1, 2]} intensity={0.5} />

      <group ref={groupRef} rotation={[0.12, 0, 0]}>
        {BARS.map((bar, i) => (
          <mesh
            key={bar.x}
            ref={(el) => {
              barRefs.current[i] = el;
            }}
            position={[bar.x, 0, 0]}
            material={material}
          >
            <boxGeometry args={[BAR_WIDTH, bar.height * 1.9, BAR_WIDTH]} />
          </mesh>
        ))}
      </group>
    </>
  );
}

export function SignalMark() {
  return (
    <Stage
      className="h-7 w-7 shrink-0"
      camera={{ position: [0, 0, 3.4], fov: 34 }}
      dpr={[1, 2]}
      fallback={<FlatMark className="h-7 w-7 p-0.5 text-accent" />}
    >
      <Bars />
    </Stage>
  );
}
