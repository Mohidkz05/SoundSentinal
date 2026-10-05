/**
 * Pulling the soundtrack out of a video, in the browser.
 *
 * The server decodes audio with libsndfile (WAV, FLAC, MP3, Ogg), which cannot
 * open MP4, MOV, WebM or M4A — those wrap AAC or Opus in a container it has
 * never heard of, and adding FFmpeg to the server needs system packages. Every
 * browser already ships those decoders behind `decodeAudioData`, so the
 * conversion happens here: decode, mix to mono, resample to the model's rate,
 * and write an uncompressed WAV. The video itself never leaves the browser.
 *
 * WAV rather than MP3 on purpose. MP3 is lossy, and the artefacts it adds are
 * exactly the kind of thing the model looks for; re-encoding would put a
 * second codec between the clip and the reading. 16-bit PCM at 16 kHz adds
 * nothing the model would not have done to the audio itself.
 */

import { MODEL_SAMPLE_RATE } from './peaks';

/** Containers the server can't read and the browser can. */
export const VIDEO_FORMATS = ['mp4', 'm4a', 'mov', 'webm'];

/** The browser holds the whole file in memory to decode it. */
export const MAX_VIDEO_BYTES = 200 * 1024 * 1024;

/** Audio kept from a video. The model reads the first 4 s; two minutes is
 *  enough for the waveform to show what was cut, and at 16-bit 16 kHz mono it
 *  comes to 3.8 MB — under the 5 MB upload limit with room to spare. */
export const MAX_EXTRACT_SECONDS = 120;

/** Mono float samples -> a 16-bit PCM WAV blob. */
function encodeWav(samples, sampleRate) {
  const bytes = 44 + samples.length * 2;
  const view = new DataView(new ArrayBuffer(bytes));
  const text = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  text(0, 'RIFF');
  view.setUint32(4, bytes - 8, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

/**
 * Decode `file`'s audio track and return it as a 16 kHz mono WAV `File`.
 * Throws when the browser can't decode it or the file has no audio track.
 *
 * @returns {Promise<{file: File, duration: number, channels: number}>}
 *          `duration` and `channels` describe the original soundtrack.
 */
export async function extractAudio(file) {
  const Ctx =
    typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext);
  if (!Ctx || typeof OfflineAudioContext === 'undefined') {
    throw new Error('Web Audio is unavailable in this browser.');
  }

  const bytes = await file.arrayBuffer();
  const ctx = new Ctx();
  let buffer;
  try {
    buffer = await ctx.decodeAudioData(bytes);
  } finally {
    ctx.close?.();
  }

  const seconds = Math.min(buffer.duration, MAX_EXTRACT_SECONDS);
  const frames = Math.max(1, Math.ceil(seconds * MODEL_SAMPLE_RATE));

  /* One output channel: Web Audio's standard down-mix averages the inputs,
     the same mono the server would have made. Rendering into a 16 kHz
     context is the resample. */
  const offline = new OfflineAudioContext(1, frames, MODEL_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();

  const base = file.name.replace(/\.[^.]+$/, '') || 'clip';
  const wav = encodeWav(rendered.getChannelData(0), MODEL_SAMPLE_RATE);
  return {
    file: new File([wav], `${base}.wav`, { type: 'audio/wav' }),
    duration: buffer.duration,
    channels: buffer.numberOfChannels,
  };
}
