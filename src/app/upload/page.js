'use client';

import React, { useCallback, useId, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../../../components/header';
import Footer from '../../../components/footer';
import { Button } from '../../../components/ui/button';
import { Stat } from '../../../components/ui/stat';
import { Notice } from '../../../components/ui/notice';
import { Stepper } from '../../../components/ui/stepper';
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
import { extractAudio, MAX_VIDEO_BYTES, VIDEO_FORMATS } from '../../lib/extract';

/* The page used to advertise "up to 5mb" and "MP3, Wav" without checking
   either. These are the numbers it now actually enforces — keep them in step
   with `MAX_CONTENT_LENGTH` on the Flask side once the API is wired. */
const MAX_BYTES = 5 * 1024 * 1024;
const FORMATS = ['wav', 'mp3', 'flac'];
/* Video (and M4A) is converted to WAV in this browser before anything is sent
   — see src/lib/extract.js — so it has its own, larger limit: what reaches the
   server is the extracted soundtrack, which always fits under MAX_BYTES. */
const ACCEPT = [
  ...[...FORMATS, ...VIDEO_FORMATS].map((f) => `.${f}`),
  'audio/wav',
  'audio/mpeg',
  'audio/flac',
  'audio/x-flac',
  'audio/mp4',
  'video/mp4',
  'video/quicktime',
  'video/webm',
].join(',');

const isVideo = (name) => VIDEO_FORMATS.includes(extensionOf(name));

function extensionOf(name) {
  const i = name.lastIndexOf('.');
  return i === -1 ? '' : name.slice(i + 1).toLowerCase();
}

/** Returns an error string, or null when the file is acceptable. Messages say
 *  what went wrong and what to do — never just "invalid file". */
function validate(file) {
  const ext = extensionOf(file.name);
  const video = VIDEO_FORMATS.includes(ext);
  if (!FORMATS.includes(ext) && !video) {
    return `${ext ? `.${ext}` : 'That file type'} isn't supported. Use ${[
      ...FORMATS,
      ...VIDEO_FORMATS,
    ]
      .map((f) => `.${f}`)
      .join(', ')}.`;
  }
  const limit = video ? MAX_VIDEO_BYTES : MAX_BYTES;
  if (file.size > limit) {
    return `That ${video ? 'video' : 'clip'} is ${formatBytes(file.size)}. The limit is ${formatBytes(
      limit
    )} — try a shorter excerpt.`;
  }
  return null;
}

export default function UploadPage() {
  const router = useRouter();
  const inputId = useId();
  const inputRef = useRef(null);
  const fieldRef = useRef(null);

  const [dragging, setDragging] = useState(false);
  /* `source` is what you chose; `file` is what gets sent. They are the same
     file for audio. For a video, `file` is the WAV extracted from it, and
     stays null until extraction has finished. */
  const [source, setSource] = useState(null);
  const [file, setFile] = useState(null);
  const [extracting, setExtracting] = useState(false);
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
      setSource(null);
      setFile(null);
      setExtracting(false);
      setAnalysis(null);
      setPreviewFailed(false);
      setDecoding(false);
      setError(problem);
      return;
    }

    setError(null);
    setSource(candidate);
    setAnalysis(null);
    setPreviewFailed(false);

    /* A video has to decode: unlike the preview, extraction is the only way
       its audio reaches the model, so a failure here is an error. */
    if (isVideo(candidate.name)) {
      setFile(null);
      setExtracting(true);
      setDecoding(false);
      extractAudio(candidate)
        .then(async (extracted) => {
          if (decodeToken.current !== token) return;
          const preview = await analyseAudioFile(extracted.file).catch(() => null);
          if (decodeToken.current !== token) return;
          setFile(extracted.file);
          // The original soundtrack's length and channels, the extract's shape.
          // Its sample rate isn't knowable from here, so it isn't shown.
          setAnalysis({
            peaks: preview?.peaks ?? null,
            duration: extracted.duration,
            channels: extracted.channels,
            sampleRate: null,
          });
          setExtracting(false);
        })
        .catch(() => {
          if (decodeToken.current !== token) return;
          setSource(null);
          setExtracting(false);
          setError(
            "This browser couldn't read an audio track from that file. It may have no sound, or use a codec the browser can't decode — try exporting the audio as MP3 or WAV."
          );
        });
      return;
    }

    setFile(candidate);
    setExtracting(false);
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
    setSource(null);
    setFile(null);
    setExtracting(false);
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
        name: source?.name ?? file.name,
        size: source?.size ?? file.size,
        extracted: source !== file,
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

          <div className="shell pb-40 pt-[calc(var(--space-section)*0.6)]">
            <div className="animate-rise flex flex-col gap-4">
              <Stepper current={0} />
              <h1 className="max-w-[18ch] text-display text-balance">
                Drop a clip on the surface.
              </h1>
              <p className="max-w-[54ch] text-body text-secondary">
                Four seconds of clear speech is enough. What you are looking at
                is the shape the model reads — frequency across, time back,
                energy up. Your clip replaces it.
              </p>

              <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-3">
                {source ? (
                  <>
                    <Button
                      variant="primary"
                      size="lg"
                      onClick={analyse}
                      loading={submitting || extracting}
                      disabled={!file}
                    >
                      {submitting ? 'Analysing…' : extracting ? 'Extracting audio…' : 'Analyse clip'}
                    </Button>
                    {/* The file, its state, and the way to change it, on one
                        raised chip. The line used to sit straight on the
                        intake surface, where the bars rose through it and took
                        its contrast — and so did the "change file" control. */}
                    <div className="panel-raised flex min-h-[var(--hit)] items-center gap-4 py-1.5 pl-4 pr-1.5">
                      <div className="flex min-w-0 flex-col gap-0.5">
                        <p className="truncate text-small font-semibold text-primary">{source.name}</p>
                        <p className="tick-label" aria-live="polite">
                          {formatBytes(source.size)} ·{' '}
                          {submitting
                            ? 'sending to the model'
                            : extracting
                              ? 'extracting audio'
                              : decoding
                                ? 'reading waveform'
                                : source !== file
                                  ? 'audio extracted · ready'
                                  : 'ready'}
                        </p>
                      </div>
                      {!submitting && (
                        <Button variant="quiet" size="sm" onClick={clear}>
                          Change file
                        </Button>
                      )}
                    </div>
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
                      {FORMATS.join(' · ')} up to {formatBytes(MAX_BYTES)} ·{' '}
                      {VIDEO_FORMATS.join(' · ')} up to {formatBytes(MAX_VIDEO_BYTES)}
                    </p>
                  </>
                )}
              </div>

              {error && (
                <Notice tone="danger" className="mt-2">
                  {error}
                </Notice>
              )}

              {previewFailed && source && (
                <Notice className="mt-2">
                  This browser couldn&apos;t decode the file for a preview —
                  often the case for FLAC outside Chrome. It can still be
                  analysed; the server decodes it separately.
                </Notice>
              )}
            </div>
          </div>

          <input
            ref={inputRef}
            id={inputId}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            /* Out of the tab order: "Choose a file" is the keyboard path to
               it, and a second, invisible stop right after it is a trap. */
            tabIndex={-1}
            onChange={(e) => {
              const picked = e.target.files?.[0];
              if (picked) accept(picked);
            }}
          />
        </div>

        {/* What we measured ---------------------------------------------- */}
        {analysis && (
          <section className="band animate-rise">
            <h2 className="text-h3">Measured from your file</h2>
            <div className="mt-[var(--space-group)] grid gap-[var(--space-group)] sm:grid-cols-2 lg:grid-cols-4">
              <Stat
                label="Duration"
                value={formatDuration(analysis.duration)}
                note={truncated}
              />
              {/* Only shown when the container header actually gave us a rate.
                  An unreadable header means we don't know it, and a guess on
                  this screen would be a made-up measurement. */}
              {analysis.sampleRate && (
                <Stat
                  label="Sample rate"
                  value={formatSampleRate(analysis.sampleRate)}
                  note={
                    needsResampling(analysis.sampleRate)
                      ? `Resampled to ${formatSampleRate(MODEL_SAMPLE_RATE)}`
                      : 'Matches the model'
                  }
                />
              )}
              <Stat
                label="Channels"
                value={String(analysis.channels)}
                note={analysis.channels > 1 ? 'Mixed down to mono' : 'Mono'}
              />
              <Stat
                label="File size"
                value={formatBytes(source?.size ?? NaN)}
                note={
                  source && file && source !== file
                    ? `${formatBytes(file.size)} of audio sent`
                    : null
                }
              />
            </div>
          </section>
        )}

        {/* Standing explanation ------------------------------------------- */}
        <section className="band grid gap-[var(--space-group)] lg:grid-cols-3 lg:gap-14">
          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">What the model reads</h2>
            <p className="text-small text-secondary">
              A fixed {MODEL_WINDOW_SECONDS}-second window of mono audio at{' '}
              {formatSampleRate(MODEL_SAMPLE_RATE)}. Anything longer is
              truncated; anything shorter is padded with silence. The clip is
              then reduced to a log-Mel spectrogram, which is the only thing the
              network ever sees.
            </p>
          </div>

          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">Where your clip goes</h2>
            <p className="text-small text-secondary">
              Clips are processed in memory and never written to disk or logged.
              The surface above was decoded in this browser and never left it.
              A video stays here too: only its soundtrack, converted to WAV in
              this browser, is sent.
            </p>
          </div>

          <div className="flex flex-col gap-[var(--space-stack)]">
            <h2 className="text-h3">What it can&apos;t do</h2>
            <p className="text-small text-secondary">
              It learned from 2019-era and open-source synthetic speech. A clip
              from a commercial cloning service it never heard, or one that is
              studio-clean, is where it is weakest — clean fakes often score
              below the threshold. The result page shows how often it is wrong.
            </p>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
