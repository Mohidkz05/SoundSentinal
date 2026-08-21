'use client';

import React, { useCallback, useId, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../../../components/header';
import { Button } from '../../../components/ui/button';
import { IntakeField } from '../../../components/three/lazy';
import { storeClip, formatBytes } from '../../lib/clip';
import {
  analyseAudioFile,
  formatDuration,
  formatSampleRate,
  needsResampling,
  MODEL_SAMPLE_RATE,
  MODEL_WINDOW_SECONDS,
} from '../../lib/peaks';

/* The page used to advertise "up to 5mb" and "MP3, Wav" without checking
   either. These are the numbers it now actually enforces — keep them in step
   with `MAX_CONTENT_LENGTH` on the Flask side once the API is wired. */
const MAX_BYTES = 5 * 1024 * 1024;
const FORMATS = ['wav', 'mp3', 'flac'];
const ACCEPT = '.wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac,audio/x-flac';

function extensionOf(name) {
  const i = name.lastIndexOf('.');
  return i === -1 ? '' : name.slice(i + 1).toLowerCase();
}

/** Returns an error string, or null when the file is acceptable. Messages say
 *  what went wrong and what to do — never just "invalid file". */
function validate(file) {
  const ext = extensionOf(file.name);
  if (!FORMATS.includes(ext)) {
    return `${ext ? `.${ext}` : 'That file type'} isn't supported. Use a ${FORMATS
      .map((f) => `.${f}`)
      .join(', ')} file.`;
  }
  if (file.size > MAX_BYTES) {
    return `That clip is ${formatBytes(file.size)}. The limit is ${formatBytes(
      MAX_BYTES
    )} — try a shorter excerpt.`;
  }
  return null;
}

/** One measured value from the decoded clip. */
function Readout({ label, value, note }) {
  return (
    <div className="flex flex-col gap-1">
      <p className="tick-label">{label}</p>
      <p className="tabular text-h3 leading-none text-primary">{value}</p>
      {note && <p className="text-small text-muted">{note}</p>}
    </div>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const inputId = useId();
  const inputRef = useRef(null);
  const fieldRef = useRef(null);

  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  /* Pointer position over the intake field, normalised to 0–1 each way with y
     measured from the top. A ref rather than state: only the render loop reads
     it, and putting a mousemove through setState would re-render this whole page
     sixty times a second to deliver a value React never displays. The canvas is
     `pointer-events-none` by design, so this element does the listening and the
     geometry reads the ref — it never intercepts a drag. */
  const pointerRef = useRef({ x: 0.5, y: 0.5, active: false });

  /* The decoded clip, used for the display, the readouts, and the handoff to
     /result. Decoding is a preview, never a gate: a browser that can't decode
     FLAC says so quietly and the file is still submittable, because the server
     does its own decoding with libsndfile and has no such gap. */
  const [analysis, setAnalysis] = useState(null);
  const [decoding, setDecoding] = useState(false);
  const [previewFailed, setPreviewFailed] = useState(false);

  // Guards against a slow decode of an abandoned file landing after a newer one.
  const decodeToken = useRef(0);

  const accept = useCallback((candidate) => {
    const problem = validate(candidate);
    decodeToken.current += 1;
    const token = decodeToken.current;

    if (problem) {
      setFile(null);
      setAnalysis(null);
      setPreviewFailed(false);
      setDecoding(false);
      setError(problem);
      return;
    }

    setError(null);
    setFile(candidate);
    setAnalysis(null);
    setPreviewFailed(false);
    setDecoding(true);

    analyseAudioFile(candidate)
      .then((result) => {
        if (decodeToken.current !== token) return;
        setAnalysis(result);
        setDecoding(false);
      })
      .catch(() => {
        if (decodeToken.current !== token) return;
        setPreviewFailed(true);
        setDecoding(false);
      });
  }, []);

  const trackPointer = useCallback((e) => {
    const box = fieldRef.current?.getBoundingClientRect();
    if (!box || box.width === 0 || box.height === 0) return;
    const p = pointerRef.current;
    p.x = Math.min(Math.max((e.clientX - box.left) / box.width, 0), 1);
    p.y = Math.min(Math.max((e.clientY - box.top) / box.height, 0), 1);
    p.active = true;
  }, []);

  const releasePointer = useCallback(() => {
    pointerRef.current.active = false;
  }, []);

  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items?.length) setDragging(true);
  }, []);

  const onDragLeave = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      releasePointer();
    },
    [releasePointer]
  );

  const onDragOver = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      /* A drag fires dragover, not pointermove, so the swell would freeze
         mid-drag without this. */
      trackPointer(e);
    },
    [trackPointer]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) accept(dropped);
    },
    [accept]
  );

  const clear = () => {
    decodeToken.current += 1;
    setFile(null);
    setAnalysis(null);
    setPreviewFailed(false);
    setDecoding(false);
    setSubmitting(false);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  /* The submit path. Posts to the Next.js proxy route rather than to Flask
     directly — see the comment at the top of src/app/api/predict/route.js.

     Nothing is stored until the model has answered, and the clip and the
     reading are written together: navigating first and fetching on /result
     would mean a page whose whole subject is a number arriving without one.
     A failure keeps you here, where the file still is. */
  const analyse = async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/predict', { method: 'POST', body });
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        setError(payload?.error ?? 'The clip could not be analysed. Try again.');
        setSubmitting(false);
        return;
      }

      storeClip({
        name: file.name,
        size: file.size,
        duration: analysis?.duration ?? null,
        sampleRate: analysis?.sampleRate ?? null,
        channels: analysis?.channels ?? null,
        peaks: analysis?.peaks ?? null,
        reading: payload,
      });
      router.push('/result');
      /* Deliberately still submitting: the navigation is in flight and the
         button should stay spent rather than flicking back to "Analyse clip"
         for the moment before the route changes. */
    } catch {
      setError('The clip could not be sent for analysis. Check your connection and try again.');
      setSubmitting(false);
    }
  };

  const truncated =
    analysis && analysis.duration > MODEL_WINDOW_SECONDS
      ? `First ${MODEL_WINDOW_SECONDS} s analysed`
      : null;

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="flex-1">
        {/* The intake field is the page, not an illustration on it. It runs
            the full bleed and the drop target is the field itself — you drop a
            clip onto the surface it is about to become, rather than onto a
            dashed rectangle that happens to sit near a picture. */}
        <div
          ref={fieldRef}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onPointerMove={trackPointer}
          onPointerLeave={releasePointer}
          className="relative isolate flex min-h-[clamp(32rem,78vh,50rem)] flex-col overflow-hidden"
        >
          <div className="absolute inset-0 -z-20">
            <IntakeField
              className="h-full w-full"
              peaks={analysis?.peaks ?? null}
              active={dragging}
              pointerRef={pointerRef}
            />
          </div>

          {/* Copy at the top, surface below it. The first arrangement put the
              text over the field and needed a wash heavy enough to bury the
              thing the page is built around — in dark mode it erased it
              entirely. Reading order and depth order agree this way round:
              you read the instruction, then look down at the instrument. The
              wash only has to cover the top band where the two still meet. */}
          <div
            className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-1/2"
            style={{
              background:
                'linear-gradient(to bottom, var(--canvas) 32%, color-mix(in oklab, var(--canvas) 78%, transparent) 68%, transparent 100%)',
            }}
            aria-hidden="true"
          />

          <div className="shell pb-40 pt-20">
            <div className="animate-rise flex flex-col gap-4">
              <p className="tick-label">Step 1 of 2</p>
              <h1 className="max-w-[18ch] text-display text-balance">
                Drop a clip on the surface.
              </h1>
              <p className="max-w-[54ch] text-body text-secondary">
                Four seconds of clear speech is enough. What you are looking at
                is the shape the model reads — frequency across, time back,
                energy up. Your clip replaces it.
              </p>

              <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-3">
                {file ? (
                  <>
                    <Button
                      variant="primary"
                      size="lg"
                      onClick={analyse}
                      loading={submitting}
                    >
                      {submitting ? 'Analysing…' : 'Analyse clip'}
                    </Button>
                    <div className="flex flex-col gap-0.5">
                      <p className="text-body font-semibold text-primary">
                        {file.name}
                      </p>
                      {/* One line, three states, in the order they happen:
                          decoding for the preview, waiting on the model, done. */}
                      <p className="tick-label" aria-live="polite">
                        {formatBytes(file.size)} ·{' '}
                        {submitting
                          ? 'sending to the model'
                          : decoding
                            ? 'reading waveform'
                            : 'ready'}
                      </p>
                    </div>
                    {!submitting && (
                      <Button variant="quiet" size="sm" onClick={clear}>
                        Choose a different file
                      </Button>
                    )}
                  </>
                ) : (
                  <>
                    {/* Opens the input rather than wrapping it in a label: a
                        button inside a label is invalid nesting, and this keeps
                        the control a real button for keyboard and AT. */}
                    <Button
                      variant="primary"
                      size="lg"
                      onClick={() => inputRef.current?.click()}
                      aria-controls={inputId}
                    >
                      Choose a file
                    </Button>
                    <p className="text-small text-secondary">
                      or drag one anywhere on this panel
                    </p>
                    <p className="tick-label">
                      {FORMATS.join(' · ')} — up to {formatBytes(MAX_BYTES)}
                    </p>
                  </>
                )}
              </div>

              {error && (
                <p
                  role="alert"
                  className="mt-2 flex items-start gap-2 text-small text-danger"
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="mt-0.5 h-4 w-4 flex-none"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="9" />
                    <path d="M12 7.5v5M12 16.2v.1" />
                  </svg>
                  {error}
                </p>
              )}

              {previewFailed && file && (
                <p className="mt-2 max-w-[62ch] text-small text-muted">
                  This browser couldn&apos;t decode the file for a preview —
                  often the case for FLAC outside Chrome. It can still be
                  analysed; the server decodes it separately.
                </p>
              )}
            </div>
          </div>

          <input
            ref={inputRef}
            id={inputId}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={(e) => {
              const picked = e.target.files?.[0];
              if (picked) accept(picked);
            }}
          />
        </div>

        {/* What we measured ---------------------------------------------- */}
        {analysis && (
          <section className="rule-full animate-rise shell py-10">
            <p className="tick-label mb-6">Measured from your file</p>
            <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
              <Readout
                label="Duration"
                value={formatDuration(analysis.duration)}
                note={truncated}
              />
              {/* Only shown when the container header actually gave us a rate.
                  An unreadable header means we don't know it, and a guess on
                  this screen would be a made-up measurement. */}
              {analysis.sampleRate && (
                <Readout
                  label="Sample rate"
                  value={formatSampleRate(analysis.sampleRate)}
                  note={
                    needsResampling(analysis.sampleRate)
                      ? `Resampled to ${formatSampleRate(MODEL_SAMPLE_RATE)}`
                      : 'Matches the model'
                  }
                />
              )}
              <Readout
                label="Channels"
                value={String(analysis.channels)}
                note={analysis.channels > 1 ? 'Mixed down to mono' : 'Mono'}
              />
              <Readout label="File size" value={formatBytes(file?.size ?? NaN)} />
            </div>
          </section>
        )}

        {/* Standing explanation ------------------------------------------- */}
        <section className="rule-full shell grid gap-10 py-12 lg:grid-cols-3">
          <div className="flex flex-col gap-3">
            <h2 className="text-h3">What the model reads</h2>
            <p className="text-small text-secondary">
              A fixed {MODEL_WINDOW_SECONDS}-second window of mono audio at{' '}
              {formatSampleRate(MODEL_SAMPLE_RATE)}. Anything longer is
              truncated; anything shorter is padded with silence. The clip is
              then reduced to a log-Mel spectrogram, which is the only thing the
              network ever sees.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <h2 className="text-h3">Where your clip goes</h2>
            <p className="text-small text-secondary">
              Clips are processed in memory and never written to disk or logged.
              The surface above was decoded in this browser and never left it.
            </p>
            <p className="text-small text-muted">
              That is a separate guarantee from the differential privacy used
              during training, which protects the corpus the model learned from.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <h2 className="text-h3">What it can&apos;t do</h2>
            <p className="text-small text-secondary">
              It was trained on one corpus of one kind of attack. A clip that is
              noisy, heavily compressed, or produced by a system newer than that
              corpus is outside what it has seen, and the reading will be worth
              less than the number suggests.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
