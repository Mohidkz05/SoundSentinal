'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import data from '../../src/lib/samples.json';
import {
  confidence,
  formatConfidence,
  formatScore,
  tierFor,
  tiersFor,
} from '../../src/lib/verdict';
import {
  useInView,
  useMotionPaused,
  usePrefersReducedMotion,
} from '../three/use-stage';
import { VerdictScale } from './verdict-scale';

/**
 * The home page's hook: a real reading, shown before anyone uploads anything.
 *
 * Two clips (one synthetic, one real) as the served model actually scored
 * them. `src/lib/samples.json` is written by `ai_model/sample_readings.py`
 * from the live API, so every number here comes from that model, and the
 * caption says the clips are from its training data. This shows what a reading
 * looks like, not how often one is right.
 *
 * One authored moment: the playhead scans the waveform, then the needle
 * travels to the new reading (on --ease-needle, inside VerdictScale). Swapping
 * samples swings the needle across the threshold, which is the product's whole
 * argument in one motion. It is DOM and CSS, not WebGL, so it is always there.
 *
 * The page cycles between the two until the visitor picks one, and only while
 * the panel is on screen, motion is allowed and the footer's pause switch is
 * off (WCAG 2.2.2).
 */
const SCAN_MS = 1100;
const CYCLE_MS = 6500;

const { threshold, bandLow, samples } = data;
const TIERS = tiersFor(threshold, bandLow);

export function SampleReading({ className = '' }) {
  const hostRef = useRef(null);
  const waveRef = useRef(null);
  const audioRef = useRef(null);
  const inView = useInView(hostRef, '0px');
  const reduced = usePrefersReducedMotion();
  const paused = useMotionPaused();

  const [index, setIndex] = useState(0);
  /* The reading on the scale trails the waveform: it changes when the scan
     finishes, so the needle moves because the clip was read. */
  const [shown, setShown] = useState(0);
  const [auto, setAuto] = useState(true);
  const [playing, setPlaying] = useState(false);

  const setScan = useCallback((p) => {
    waveRef.current?.style.setProperty('--scan', String(p));
  }, []);

  // The scan, on every change of sample. Written to a CSS variable, not state.
  useEffect(() => {
    if (reduced) {
      setScan(1);
      setShown(index);
      return;
    }
    let raf;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / SCAN_MS, 1);
      setScan(1 - Math.pow(1 - t, 3));
      if (t < 1) raf = requestAnimationFrame(step);
      else setShown(index);
    };
    setScan(0);
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [index, reduced, setScan]);

  // While the clip plays, the playhead follows the audio instead.
  useEffect(() => {
    if (!playing) return;
    let raf;
    const follow = () => {
      const a = audioRef.current;
      if (a && a.duration) setScan(a.currentTime / a.duration);
      raf = requestAnimationFrame(follow);
    };
    raf = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(raf);
  }, [playing, setScan]);

  // Alternate until the visitor takes over.
  useEffect(() => {
    if (!auto || !inView || paused || reduced || playing) return;
    const id = setTimeout(() => setIndex((i) => (i + 1) % samples.length), CYCLE_MS);
    return () => clearTimeout(id);
  }, [auto, inView, paused, reduced, playing, index]);

  const choose = (i) => {
    setAuto(false);
    if (i === index) return;
    audioRef.current?.pause();
    setIndex(i);
  };

  const toggleAudio = () => {
    const a = audioRef.current;
    if (!a) return;
    setAuto(false);
    if (playing) {
      a.pause();
    } else {
      a.currentTime = 0;
      a.play().catch(() => setPlaying(false));
    }
  };

  const sample = samples[index];
  const reading = samples[shown];
  const tier = tierFor(reading.score, TIERS);
  const flagged = reading.score >= threshold;

  return (
    <figure ref={hostRef} className={`panel-raised p-5 sm:p-7 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="tick-label" id="sample-label">Sample reading</p>
        <div role="group" aria-labelledby="sample-label" className="flex gap-1 rounded-[var(--radius-md)] border border-line p-1">
          {samples.map((s, i) => (
            <button
              key={s.id}
              type="button"
              aria-pressed={i === index}
              onClick={() => choose(i)}
              className={`min-h-[var(--hit)] rounded-[var(--radius-sm)] px-3.5 text-small transition-colors duration-[var(--duration-fast)] ease-[var(--ease-instrument)] ${
                i === index
                  ? 'bg-overlay font-semibold text-primary'
                  : 'text-secondary hover:text-primary'
              }`}
            >
              {s.title}
            </button>
          ))}
        </div>
      </div>

      {/* The clip. Teal, because it is the input, not a result (DESIGN.md
          rule 1); the scanned part is drawn by clipping a second copy, so the
          bars never re-render while the playhead moves. */}
      <div className="mt-6 flex items-center gap-4">
        <button
          type="button"
          onClick={toggleAudio}
          aria-label={`${playing ? 'Stop' : 'Play'} the ${sample.title.toLowerCase()} sample`}
          className="grid h-11 w-11 flex-none place-items-center rounded-full border border-line-strong text-primary transition-colors duration-[var(--duration-fast)] hover:border-accent hover:text-accent"
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            {playing ? (
              <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" />
            ) : (
              <path d="M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2A.8.8 0 0 0 8 5.8Z" />
            )}
          </svg>
        </button>
        <div ref={waveRef} className="relative h-16 flex-1" style={{ '--scan': 1 }} aria-hidden="true">
          <Bars peaks={sample.peaks} className="text-line-strong" />
          <Bars
            peaks={sample.peaks}
            className="text-accent"
            style={{ clipPath: 'inset(0 calc((1 - var(--scan)) * 100%) 0 0)' }}
          />
          <div
            className="absolute inset-y-[-4px] w-px bg-primary"
            style={{ left: 'calc(var(--scan) * 100%)', opacity: 'calc((1 - var(--scan)) * 40 )' }}
          />
        </div>
        <audio
          ref={audioRef}
          src={sample.src}
          preload="none"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            setPlaying(false);
            setScan(1);
          }}
        />
      </div>

      {/* The reading. Three channels, as on /result: number, tier name,
          needle position. Hue is the tier's, never the brand's. */}
      <div className="mt-7" aria-live={auto ? 'off' : 'polite'}>
        <p className="tick-label">Model confidence it is {flagged ? 'AI generated' : 'real'}</p>
        <div className="mt-2 flex flex-wrap items-end gap-x-5 gap-y-1">
          <output
            className="font-mono text-readout font-medium tabular"
            style={{ color: tier.token }}
          >
            {formatConfidence(confidence(reading.score, threshold))}
          </output>
          <p className="pb-1.5 text-body font-semibold" style={{ color: tier.token }}>
            {tier.headline}
          </p>
        </div>
        <p className="tick-label tabular mt-2">
          Score {formatScore(reading.score)} · threshold {formatScore(threshold)}
        </p>
      </div>

      <div className="mt-6">
        <VerdictScale score={reading.score} threshold={threshold} bandLow={bandLow} height="h-10" compact />
      </div>

      <figcaption className="mt-6 text-small text-muted">
        Two clips from ASVspoof 2019, scored by the live model. It trained on
        them, so this shows what a reading looks like, not how often one is
        right.
      </figcaption>
    </figure>
  );
}

function Bars({ peaks, className = '', style }) {
  return (
    <div className={`absolute inset-0 flex items-center gap-[2px] ${className}`} style={style}>
      {peaks.map((p, i) => (
        <span
          key={i}
          className="flex-1 rounded-full bg-current"
          style={{ height: `${Math.max(p, 0.06) * 100}%` }}
        />
      ))}
    </div>
  );
}
