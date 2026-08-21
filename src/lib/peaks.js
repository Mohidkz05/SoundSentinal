/**
 * Reading a dropped clip well enough to draw it.
 *
 * The upload screen shows the waveform of the file you actually chose rather
 * than a stock animation. That is worth the decode: it confirms the browser
 * could read the file at all, and it surfaces the three numbers that decide how
 * the clip will be treated — duration, sample rate, channel count — before
 * anything is sent anywhere.
 *
 * Everything here is best-effort. `decodeAudioData` codec support varies by
 * browser (FLAC in particular is fine in Chrome and patchy elsewhere), so a
 * failure is expected, not exceptional, and the caller falls back to an idle
 * animation with nothing else lost.
 */

/** What `preprocess_waveform()` in ai_model/model.py normalises every clip to. */
export const MODEL_SAMPLE_RATE = 16000;
export const MODEL_WINDOW_SECONDS = 4;

/* ---------------------------------------------------------------------------
   Container sample rate.

   `decodeAudioData` resamples to the AudioContext's rate, so the decoded
   buffer's `sampleRate` is the output device's rate — typically 44.1 kHz —
   and not the file's. Reporting it as the clip's sample rate is simply wrong:
   the 16 kHz ASVspoof samples in this repo come back claiming 44.1 kHz.

   The rate is in the container header in every format we accept, so read it
   from there. Returns null when it can't be determined, and the UI then shows
   nothing rather than a number it made up.
   ------------------------------------------------------------------------ */

/** RIFF/WAVE: walk the chunk list to `fmt `, whose 5th–8th bytes are the rate. */
function wavSampleRate(view) {
  if (view.byteLength < 44) return null;
  if (view.getUint32(0, false) !== 0x52494646) return null; // "RIFF"
  if (view.getUint32(8, false) !== 0x57415645) return null; // "WAVE"

  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = view.getUint32(offset, false);
    const size = view.getUint32(offset + 4, true);
    if (id === 0x666d7420) {
      // "fmt ": format(2) channels(2) sampleRate(4)
      return offset + 16 <= view.byteLength ? view.getUint32(offset + 12, true) : null;
    }
    offset += 8 + size + (size % 2); // chunks are word-aligned
  }
  return null;
}

/** FLAC: STREAMINFO is always the first metadata block; the rate is 20 bits
 *  starting 10 bytes into it. */
function flacSampleRate(view) {
  if (view.byteLength < 30) return null;
  if (view.getUint32(0, false) !== 0x664c6143) return null; // "fLaC"
  if ((view.getUint8(4) & 0x7f) !== 0) return null; // block type 0 = STREAMINFO

  const streamInfo = 8;
  const a = view.getUint8(streamInfo + 10);
  const b = view.getUint8(streamInfo + 11);
  const c = view.getUint8(streamInfo + 12);
  const rate = (a << 12) | (b << 4) | (c >> 4);
  return rate > 0 ? rate : null;
}

const MPEG_RATES = {
  0: [11025, 12000, 8000], // MPEG 2.5
  2: [22050, 24000, 16000], // MPEG 2
  3: [44100, 48000, 32000], // MPEG 1
};

/** MP3: skip any ID3v2 tag, then find the first frame sync and read its rate. */
function mp3SampleRate(view) {
  let offset = 0;

  if (
    view.byteLength > 10 &&
    view.getUint8(0) === 0x49 && // 'I'
    view.getUint8(1) === 0x44 && // 'D'
    view.getUint8(2) === 0x33 // '3'
  ) {
    // ID3v2 size is four synchsafe bytes: 7 significant bits each.
    const size =
      ((view.getUint8(6) & 0x7f) << 21) |
      ((view.getUint8(7) & 0x7f) << 14) |
      ((view.getUint8(8) & 0x7f) << 7) |
      (view.getUint8(9) & 0x7f);
    offset = 10 + size;
  }

  const limit = Math.min(view.byteLength - 3, offset + 200000);
  for (let i = Math.max(0, offset); i < limit; i++) {
    if (view.getUint8(i) !== 0xff) continue;
    const b1 = view.getUint8(i + 1);
    if ((b1 & 0xe0) !== 0xe0) continue; // 11 sync bits

    const version = (b1 >> 3) & 0x03;
    const rates = MPEG_RATES[version];
    if (!rates) continue; // 1 is reserved

    const index = (view.getUint8(i + 2) >> 2) & 0x03;
    if (index === 3) continue; // reserved
    return rates[index];
  }
  return null;
}

function containerSampleRate(bytes) {
  const view = new DataView(bytes);
  try {
    return wavSampleRate(view) ?? flacSampleRate(view) ?? mp3SampleRate(view);
  } catch {
    return null; // A truncated or unexpected header is not worth failing over.
  }
}

/**
 * Decode `file` and reduce it to `buckets` RMS envelope values in 0–1.
 *
 * RMS rather than peak amplitude: peak picking on a sparse bucket grid latches
 * onto single-sample transients and draws a spiky, misleading envelope.
 *
 * `sampleRate` comes from the container header and is null when unreadable —
 * never from the decoded buffer, which reports the output device's rate.
 *
 * @returns {Promise<{peaks: Float32Array, duration: number, sampleRate: number|null, channels: number}>}
 */
export async function analyseAudioFile(file, buckets = 320) {
  const Ctx =
    typeof window !== 'undefined' &&
    (window.AudioContext || window.webkitAudioContext);
  if (!Ctx) throw new Error('Web Audio is unavailable in this browser.');

  const bytes = await file.arrayBuffer();
  // Read the header before decoding: `decodeAudioData` detaches the buffer.
  const sampleRate = containerSampleRate(bytes);
  const ctx = new Ctx();

  try {
    const buffer = await ctx.decodeAudioData(bytes);
    const channels = buffer.numberOfChannels;
    const length = buffer.length;

    // Downmix to mono, which is what the model reads anyway.
    const mono = new Float32Array(length);
    for (let c = 0; c < channels; c++) {
      const data = buffer.getChannelData(c);
      for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
    }

    const peaks = new Float32Array(buckets);
    const per = Math.max(1, Math.floor(length / buckets));
    let max = 0;

    for (let b = 0; b < buckets; b++) {
      const start = b * per;
      const end = Math.min(length, start + per);
      let sum = 0;
      for (let i = start; i < end; i++) sum += mono[i] * mono[i];
      const rms = end > start ? Math.sqrt(sum / (end - start)) : 0;
      peaks[b] = rms;
      if (rms > max) max = rms;
    }

    // Normalise so a quiet recording still draws at full height — the picture
    // is of the clip's shape, not its absolute level.
    if (max > 0) for (let b = 0; b < buckets; b++) peaks[b] /= max;

    return { peaks, duration: buffer.duration, sampleRate, channels };
  } finally {
    // Contexts are a limited resource and this one has done its job.
    ctx.close?.();
  }
}

/** `3.42 s`, or `—` when the duration is unknown. */
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  return `${seconds.toFixed(2)} s`;
}

/** `44.1 kHz` */
export function formatSampleRate(hz) {
  if (!Number.isFinite(hz)) return '—';
  return `${(hz / 1000).toFixed(1)} kHz`;
}

/** True when the model will have to resample this clip. */
export function needsResampling(hz) {
  return Number.isFinite(hz) && hz !== MODEL_SAMPLE_RATE;
}
