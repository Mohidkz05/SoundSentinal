'use client';

import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../../../components/header';
import Footer from '../../../components/footer';
import { Button } from '../../../components/ui/button';
import { Stat } from '../../../components/ui/stat';
import { Notice } from '../../../components/ui/notice';
import { Stepper } from '../../../components/ui/stepper';
import { ListeningWindow } from '../../../components/ui/listening-window';
import { storeClip, formatBytes } from '../../lib/clip';
import {
  analyseAudioFile,
  formatDuration,
  formatSampleRate,
  needsResampling,
  MODEL_SAMPLE_RATE,
  MODEL_WINDOW_SECONDS,
} from '../../lib/peaks';
import { extractAudio, MAX_EXTRACT_SECONDS, MAX_VIDEO_BYTES, VIDEO_FORMATS } from '../../lib/extract';
import { canRecord, useRecorder } from '../../lib/recorder';

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

/* 'webm' → 'WebM', the rest upper-case: how people write them. */
const formatName = (f) => (f === 'webm' ? 'WebM' : f.toUpperCase());
const listFormats = (list) => list.map(formatName).join(', ');
/* The limits are whole megabytes; formatBytes would say '5.0 MB'. */
const wholeMB = (bytes) => `${Math.round(bytes / (1024 * 1024))} MB`;

/* Shorter than this is a mis-click, not a clip. */
const MIN_RECORDING_SECONDS = 1;

const TIPS = [
  'Use the original file if you have it. Re-recording through a speaker or a phone call changes the sound, and the reading with it.',
  'Clean, studio-quality audio is the hardest case. A low reading on a clean clip is weak evidence that it is real.',
];

const ICON = {
  viewBox: '0 0 20 20',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

function UploadIcon({ className = 'h-4 w-4' }) {
  return (
    <svg {...ICON} className={className}>
      <path d="M10 13V3.5M6 7.5l4-4 4 4M4 13.5v2a1.5 1.5 0 0 0 1.5 1.5h9a1.5 1.5 0 0 0 1.5-1.5v-2" />
    </svg>
  );
}

function DropIcon() {
  return (
    <svg {...ICON} className="h-6 w-6">
      <path d="M10 3.5V13M6 9l4 4 4-4M4 13.5v2a1.5 1.5 0 0 0 1.5 1.5h9a1.5 1.5 0 0 0 1.5-1.5v-2" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg {...ICON} className="h-4 w-4">
      <rect x="7" y="2.5" width="6" height="9.5" rx="3" />
      <path d="M4.5 9.5a5.5 5.5 0 0 0 11 0M10 15v2.5" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <rect x="4" y="4" width="12" height="12" rx="2" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg {...ICON} className="h-4 w-4">
      <path d="M11.5 2.5H6A1.5 1.5 0 0 0 4.5 4v12A1.5 1.5 0 0 0 6 17.5h8a1.5 1.5 0 0 0 1.5-1.5V6.5l-4-4Z" />
      <path d="M11.5 2.5v4h4M8 13v-2M10 14V9.5M12 13v-2" />
    </svg>
  );
}

/** Seconds → '0:03.2', the take's running time. */
function formatTake(seconds) {
  const m = Math.floor(seconds / 60);
  const rest = seconds - m * 60;
  return `${m}:${rest.toFixed(1).padStart(4, '0')}`;
}

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

  /* `pageDrag`: a file is being dragged anywhere over the window, so the deck
     announces itself as the place to put it. `dragging`: it is over the deck. */
  const [pageDrag, setPageDrag] = useState(false);
  const [dragging, setDragging] = useState(false);
  /* `source` is what you chose; `file` is what gets sent. They are the same
     file for audio. For a video or a recording, `file` is the WAV extracted
     from it, and stays null until extraction has finished. */
  const [source, setSource] = useState(null);
  /** 'file' | 'mic' — where `source` came from. */
  const [origin, setOrigin] = useState('file');
  const [file, setFile] = useState(null);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  /* The hosted model sleeps when idle; the first analysis after a quiet spell
     waits for it to start. Past a few seconds, say so. */
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!submitting) return setSlow(false);
    const t = setTimeout(() => setSlow(true), 5000);
    return () => clearTimeout(t);
  }, [submitting]);

  /* Recording needs MediaRecorder and a secure context. Known only after
     mount, so the server render and the first client render agree. */
  const [micAvailable, setMicAvailable] = useState(false);
  useEffect(() => setMicAvailable(canRecord()), []);

  /* The decoded clip, used for the display, the readouts, and the handoff to
     /result. Decoding is a preview, never a gate: a browser that can't decode
     FLAC says so quietly and the file is still submittable, because the server
     does its own decoding with libsndfile and has no such gap. */
  const [analysis, setAnalysis] = useState(null);
  const [decoding, setDecoding] = useState(false);
  const [previewFailed, setPreviewFailed] = useState(false);

  // Guards against a slow decode of an abandoned file landing after a newer one.
  const decodeToken = useRef(0);

  const accept = useCallback((candidate, from = 'file') => {
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
    setOrigin(from);
    setAnalysis(null);
    setPreviewFailed(false);

    /* A video (or a recording, which arrives as WebM or MP4) has to decode:
       unlike the preview, extraction is the only way its audio reaches the
       model, so a failure here is an error. */
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
            from === 'mic'
              ? "This browser couldn't decode its own recording. Try recording again, or record with another app and choose the file."
              : "This browser couldn't read an audio track from that file. It may have no sound, or use a codec the browser can't decode — try exporting the audio as MP3 or WAV."
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

  const recorder = useRecorder({
    maxSeconds: MAX_EXTRACT_SECONDS,
    onDone: (recording, seconds) => {
      if (seconds < MIN_RECORDING_SECONDS) {
        setError(
          `That recording was under ${MIN_RECORDING_SECONDS} second. Speak for at least ${MODEL_WINDOW_SECONDS} seconds — that is how much the model hears.`
        );
        return;
      }
      accept(recording, 'mic');
    },
    onError: setError,
  });
  const recording = recorder.state !== 'idle';

  /* Each control here unmounts the moment it is used, so focus is moved to
     the next thing to press rather than dropped onto <body>. */
  const chooseRef = useRef(null);
  const recordRef = useRef(null);
  const stopRef = useRef(null);
  const analyseRef = useRef(null);
  const refocus = (ref) => requestAnimationFrame(() => ref.current?.focus());
  useEffect(() => {
    if (recorder.state === 'recording') stopRef.current?.focus();
  }, [recorder.state]);
  useEffect(() => {
    if (origin === 'mic' && file) analyseRef.current?.focus();
  }, [origin, file]);

  /* Busy: a drop now would land on top of a send or a take in progress. */
  const busy = useRef(false);
  busy.current = submitting || recording;

  /* The whole window takes the drop. Without this, a file released a few
     pixels off the deck makes the browser navigate away to play it — the page,
     and anything chosen on it, gone. */
  useEffect(() => {
    let depth = 0;
    const carriesFiles = (e) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
    const enter = (e) => {
      if (!carriesFiles(e)) return;
      depth += 1;
      setPageDrag(true);
    };
    const leave = (e) => {
      if (!carriesFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) {
        setPageDrag(false);
        setDragging(false);
      }
    };
    const over = (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = busy.current ? 'none' : 'copy';
    };
    const drop = (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setPageDrag(false);
      setDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped && !busy.current) accept(dropped);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', over);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', over);
      window.removeEventListener('drop', drop);
    };
  }, [accept]);

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
    setOrigin('file');
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
        name: origin === 'mic' ? 'Microphone recording' : source?.name ?? file.name,
        size: source?.size ?? file.size,
        extracted: origin !== 'mic' && source !== file,
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

  /* What the deck draws: the live take while recording, else the clip. */
  const shown = recording ? recorder.levels : analysis;
  const elapsed = recorder.levels?.duration ?? 0;
  const empty = !source && !recording;

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main id="main" className="flex-1">
        <div className="shell pb-[var(--space-section)] pt-[calc(var(--space-section)*0.6)]">
          <div className="animate-rise flex flex-col gap-4">
            <Stepper current={0} />
            {/* The headline follows the state, so the screen always says
                what to do next: choose, then analyse, then wait. */}
            <h1 className="max-w-[22ch] text-display text-balance">
              {submitting
                ? 'Reading your clip.'
                : recording
                  ? 'Listening.'
                  : source
                    ? 'Ready to analyse.'
                    : 'Drop in a clip, or record one.'}
            </h1>
            <p className="max-w-[58ch] text-lead text-secondary">
              A voice note, call recording or video{micAvailable ? ', or your own voice' : ''}.
              The model listens to the first {MODEL_WINDOW_SECONDS} seconds — the
              lit window below — and nothing is stored.
            </p>
          </div>

          {/* The intake deck (10 October 2026). One framed surface that is
              obviously the place a clip goes: the timeline of what the model
              hears, with the way in — drop, choose or record — on top of it
              until a clip arrives. Dragging a file anywhere over the page
              outlines it; over the deck itself, it lights. */}
          <div
            onDragEnter={() => setDragging(true)}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false);
            }}
            className={`intake-deck mt-[var(--space-head)] ${
              dragging ? 'is-over' : pageDrag && !(submitting || recording) ? 'is-armed' : ''
            }`}
          >
            <div className="relative">
              <ListeningWindow
                peaks={shown?.peaks ?? null}
                duration={shown?.duration ?? null}
                active={dragging}
                live={recording}
                dim={empty}
              />

              {empty && (
                <div className="intake-prompt">
                  <span className="intake-glyph" aria-hidden="true">
                    {/* Over the deck the arrow turns to point into the tray:
                        "put it here", not "send it off". */}
                    {dragging ? <DropIcon /> : <UploadIcon className="h-6 w-6" />}
                  </span>
                  <p className="text-h3 font-semibold text-primary" aria-live="polite">
                    {dragging ? 'Release to load it' : 'Drop an audio or video file here'}
                  </p>
                  <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2">
                    {/* Opens the input rather than wrapping it in a label: a
                        button inside a label is invalid nesting, and this keeps
                        the control a real button for keyboard and AT. */}
                    <Button
                      ref={chooseRef}
                      variant="primary"
                      size="lg"
                      onClick={() => inputRef.current?.click()}
                      aria-controls={inputId}
                    >
                      <UploadIcon className="h-4 w-4" />
                      Choose a file
                    </Button>
                    {micAvailable && (
                      <>
                        <span className="text-small text-muted">or</span>
                        <Button
                          ref={recordRef}
                          variant="secondary"
                          size="lg"
                          onClick={() => {
                            setError(null);
                            recorder.start();
                          }}
                        >
                          <MicIcon />
                          Record
                        </Button>
                      </>
                    )}
                  </div>
                  <p className="text-small text-muted">
                    {listFormats(FORMATS)} up to {wholeMB(MAX_BYTES)} ·{' '}
                    {listFormats(VIDEO_FORMATS)} up to {wholeMB(MAX_VIDEO_BYTES)}
                  </p>
                </div>
              )}

            </div>

            {recording && (
              <div className="intake-strip">
                <div className="flex min-w-0 items-center gap-4">
                  <span
                    className={`h-3 w-3 flex-none rounded-full bg-accent ${
                      recorder.state === 'recording' ? 'recording-dot' : 'opacity-40'
                    }`}
                    aria-hidden="true"
                  />
                  <span className="tabular w-[6ch] flex-none text-h3 text-primary">{formatTake(elapsed)}</span>
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <p className="text-small font-semibold text-primary" aria-live="polite">
                      {recorder.state === 'requesting'
                        ? 'Waiting for microphone permission'
                        : elapsed < MODEL_WINDOW_SECONDS
                          ? 'Speak — the lit window fills first'
                          : 'The model has its 4 seconds. Stop when you are done.'}
                    </p>
                    <p className="text-small text-secondary">
                      A clip played through a speaker changes its sound; the original file reads better.
                    </p>
                  </div>
                </div>
                <div className="flex flex-none items-center gap-2">
                  <Button
                    variant="quiet"
                    size="md"
                    onClick={() => {
                      recorder.cancel();
                      refocus(recordRef);
                    }}
                  >
                    Discard
                  </Button>
                  <Button
                    ref={stopRef}
                    variant="primary"
                    size="md"
                    onClick={recorder.stop}
                    disabled={recorder.state !== 'recording'}
                  >
                    <StopIcon />
                    Stop
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="mt-[var(--space-group)] grid items-start gap-x-16 gap-y-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,30rem)]">
            <div className="flex flex-col items-start gap-4">
              {source && (
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    ref={analyseRef}
                    variant="primary"
                    size="lg"
                    onClick={analyse}
                    loading={submitting || extracting}
                    disabled={!file}
                  >
                    {submitting
                      ? 'Analysing…'
                      : extracting
                        ? origin === 'mic'
                          ? 'Preparing recording…'
                          : 'Extracting audio…'
                        : 'Analyse clip'}
                  </Button>
                  {/* The clip, its state, and the way to change it, on one chip. */}
                  <div className="flex min-h-[var(--hit)] max-w-full items-center gap-4 rounded-[var(--radius-md)] border border-line bg-raised py-1.5 pl-4 pr-1.5 shadow-[var(--shadow-panel)]">
                    <span className="flex-none text-accent" aria-hidden="true">
                      {origin === 'mic' ? <MicIcon /> : <FileIcon />}
                    </span>
                    <div className="flex min-w-0 flex-col gap-0.5">
                      <p className="truncate text-small font-semibold text-primary">
                        {origin === 'mic' ? 'Microphone recording' : source.name}
                      </p>
                      <p className="text-small text-secondary">
                        {origin === 'mic' && analysis
                          ? formatDuration(analysis.duration)
                          : formatBytes(source.size)}{' '}
                        ·{' '}
                        {submitting
                          ? 'Sending to the model'
                          : extracting
                            ? origin === 'mic'
                              ? 'Preparing'
                              : 'Extracting audio'
                            : decoding
                              ? 'Reading waveform'
                              : origin === 'mic'
                                ? 'Recorded, ready'
                                : source !== file
                                  ? 'Audio extracted, ready'
                                  : 'Ready'}
                      </p>
                    </div>
                    {!submitting && (
                      <Button
                        variant="quiet"
                        size="sm"
                        onClick={() => {
                          const fromMic = origin === 'mic';
                          clear();
                          refocus(fromMic && micAvailable ? recordRef : chooseRef);
                        }}
                      >
                        {origin === 'mic' ? 'Start over' : 'Change file'}
                      </Button>
                    )}
                  </div>
                </div>
              )}

              {/* The wait is the most important thing on screen while it
                  lasts, so it gets body text in the primary ink. */}
              <p
                aria-live="polite"
                className={`flex max-w-[52ch] items-center gap-2.5 text-body text-primary ${
                  submitting && slow ? '' : 'sr-only'
                }`}
              >
                {submitting && slow && (
                  <span className="h-2 w-2 flex-none animate-pulse rounded-full bg-accent" aria-hidden="true" />
                )}
                {submitting && slow
                  ? 'Waking the model server. After a quiet spell the first reading takes up to a minute.'
                  : ''}
              </p>

              {error && <Notice tone="danger">{error}</Notice>}

              {previewFailed && source && (
                <Notice>
                  This browser couldn&apos;t decode the file for a preview —
                  often the case for FLAC outside Chrome. It can still be
                  analysed; the server decodes it separately.
                </Notice>
              )}
            </div>

            {/* The two caveats the window cannot draw. */}
            <div className={source ? '' : 'lg:col-span-2'}>
              <h2 className="field-label">For a reading worth more</h2>
              {/* Beside the clip once one is chosen; two columns under the
                  empty deck, where a lone right-hand column read as stranded. */}
              <ul className={`mt-3 grid gap-3 ${source ? '' : 'lg:grid-cols-2 lg:gap-x-16'}`}>
                {TIPS.map((tip) => (
                  <li key={tip} className="flex gap-2.5 text-small text-secondary">
                    <svg viewBox="0 0 20 20" className="mt-[0.2em] h-4 w-4 flex-none text-accent" fill="none"
                         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m4.5 10.5 3.5 3.5 7.5-8" />
                    </svg>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <input
            ref={inputRef}
            id={inputId}
            type="file"
            aria-label="Audio or video file to analyse"
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

      </main>

      <Footer />
    </div>
  );
}
