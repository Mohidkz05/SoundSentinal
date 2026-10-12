'use client';

/**
 * Recording a clip from the microphone, in the browser.
 *
 * The recording becomes an ordinary upload: MediaRecorder writes WebM/Opus
 * (Chrome, Firefox) or MP4/AAC (Safari), and the page hands that file to the
 * same `extractAudio()` path a video takes — decode, mono, 16 kHz WAV. The
 * server never learns the clip came from a microphone.
 *
 * The browser's voice processing is switched off. Echo cancellation, noise
 * suppression and automatic gain are signal processing — Chrome's noise
 * suppressor is itself a neural network — and processing is exactly what the
 * model looks for. Left on, every recording would arrive pre-altered.
 *
 * While recording, `levels` carries a growing RMS envelope (normalised to the
 * loudest moment so far) and the elapsed time, in the shape the timeline
 * draws a decoded file from, so speech appears on the ruler as it is spoken.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

/** Containers `extractAudio()` can take, in order of preference. */
const TYPES = [
  ['audio/webm;codecs=opus', 'webm'],
  ['audio/webm', 'webm'],
  ['audio/mp4', 'mp4'],
];

/** Envelope samples per second. 20 is smooth on the ruler and cheap to draw. */
const RATE = 20;

export function canRecord() {
  return (
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    TYPES.some(([type]) => MediaRecorder.isTypeSupported(type))
  );
}

/**
 * @param {{ maxSeconds: number, onDone: (file: File, seconds: number) => void, onError: (message: string) => void }} options
 */
export function useRecorder({ maxSeconds, onDone, onError }) {
  /** 'idle' | 'requesting' | 'recording' */
  const [state, setState] = useState('idle');
  const [levels, setLevels] = useState(null);
  const session = useRef(null);
  // The callbacks change identity on every render; the session outlives that.
  const done = useRef(onDone);
  const fail = useRef(onError);
  done.current = onDone;
  fail.current = onError;

  const teardown = useCallback(() => {
    const s = session.current;
    if (!s) return;
    clearInterval(s.timer);
    s.stream.getTracks().forEach((track) => track.stop());
    s.context.close().catch(() => {});
    session.current = null;
  }, []);

  const stop = useCallback(() => {
    const s = session.current;
    if (!s || s.recorder.state === 'inactive') return;
    s.recorder.stop();
  }, []);

  /** Throw the recording away, e.g. when the page is left mid-take. */
  const cancel = useCallback(() => {
    const s = session.current;
    if (!s) return;
    s.discard = true;
    if (s.recorder.state !== 'inactive') s.recorder.stop();
    teardown();
    setState('idle');
    setLevels(null);
  }, [teardown]);

  const start = useCallback(async () => {
    if (session.current) return;
    setState('requesting');

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    } catch (e) {
      setState('idle');
      fail.current(
        e?.name === 'NotAllowedError'
          ? 'Microphone access was blocked. Allow it from the icon in the address bar, or choose a file instead.'
          : e?.name === 'NotFoundError'
            ? 'No microphone was found. Plug one in, or choose a file instead.'
            : 'The microphone could not be started. Close other apps using it, or choose a file instead.'
      );
      return;
    }

    const [type, ext] = TYPES.find(([t]) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(stream, { mimeType: type });
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    context.createMediaStreamSource(stream).connect(analyser);
    const buffer = new Float32Array(analyser.fftSize);

    const chunks = [];
    const raw = [];
    let loudest = 0;
    const began = performance.now();
    const elapsed = () => (performance.now() - began) / 1000;

    const s = { stream, recorder, context, discard: false, timer: 0 };
    session.current = s;

    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const seconds = elapsed();
      teardown();
      setState('idle');
      setLevels(null);
      if (s.discard) return;
      const blob = new Blob(chunks, { type: recorder.mimeType || type });
      done.current(new File([blob], `recording.${ext}`, { type: blob.type }), seconds);
    };

    s.timer = setInterval(() => {
      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      raw.push(rms);
      // A floor, so a silent room is drawn flat rather than amplified to full.
      loudest = Math.max(loudest, rms, 0.02);
      const seconds = elapsed();
      setLevels({ peaks: raw.map((v) => v / loudest), duration: seconds });
      if (seconds >= maxSeconds) stop();
    }, 1000 / RATE);

    recorder.start(250);
    setState('recording');
  }, [maxSeconds, stop, teardown]);

  // Leaving the page mid-take releases the microphone.
  useEffect(() => cancel, [cancel]);

  return { state, levels, start, stop, cancel };
}
