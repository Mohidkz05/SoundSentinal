/**
 * Carrying the analysed clip from /upload to /result.
 *
 * The result page used to know one thing about the clip — its filename — which
 * is why it had nothing to show but the number. Everything `analyseAudioFile`
 * measured was thrown away at the page boundary: the envelope, the duration,
 * the sample rate, the channel count. All of it is worth showing next to a
 * reading, because it is the difference between "the model said 0.63" and "the
 * model said 0.63 about *this* four-second mono clip, of which it read the
 * first four seconds".
 *
 * `sessionStorage` rather than a query string or a store: it survives the
 * navigation and a refresh, it is scoped to the tab, and it is cleared when the
 * tab closes — which matches what the upload page promises about the clip not
 * being kept. It is the right shape for a handoff that becomes a POST response
 * the moment the API is wired.
 *
 * The envelope is quantised to two decimals on the way in. At 320 buckets that
 * is the difference between roughly 6 kB and 2 kB of JSON, and it is well below
 * what a bar a few pixels wide can express.
 *
 * `reading` is the model's answer, stored alongside the clip it describes so
 * the two cannot be separated — a probability shown next to the wrong waveform
 * would be worse than showing no waveform at all. It is the `/api/predict`
 * response verbatim, which is why /result reads `spoof_probability` and
 * `threshold` in the API's own naming rather than a translated copy.
 */

const KEY = 'soundsentinal-clip';

/**
 * @param {object}  clip
 * @param {string}  clip.name
 * @param {number}  clip.size            bytes
 * @param {?number} clip.duration        seconds, or null if the decode failed
 * @param {?number} clip.sampleRate      Hz from the container header, or null
 * @param {?number} clip.channels
 * @param {?Float32Array} clip.peaks     normalised envelope, or null
 * @param {?object} clip.reading          the /api/predict response, or null
 */
export function storeClip({ name, size, duration, sampleRate, channels, peaks, reading }) {
  try {
    sessionStorage.setItem(
      KEY,
      JSON.stringify({
        name,
        size,
        duration: duration ?? null,
        sampleRate: sampleRate ?? null,
        channels: channels ?? null,
        peaks: peaks ? Array.from(peaks, (v) => Math.round(v * 100) / 100) : null,
        reading: reading ?? null,
      })
    );
  } catch {
    // Storage unavailable or over quota. The result page falls back to the
    // generic labels; losing the preview is not worth blocking the analysis.
  }
}

/**
 * @returns {?object} the stored clip with `peaks` back as a Float32Array, or
 *                    null when nothing was stored or it can't be parsed.
 */
export function readClip() {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const clip = JSON.parse(raw);
    return {
      ...clip,
      peaks: Array.isArray(clip.peaks) ? Float32Array.from(clip.peaks) : null,
      /* A stored clip from before a reading existed, or one whose analysis
         failed, has no reading. /result renders its waiting state rather than
         inventing a number. */
      reading: typeof clip.reading?.spoof_probability === 'number' ? clip.reading : null,
    };
  } catch {
    return null;
  }
}

/** `48 KB`, `1.4 MB` */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
