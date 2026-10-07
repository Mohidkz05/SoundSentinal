# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**The everyday public** (confirmed 5 October 2026): someone who has received
a voice note, call recording or video clip that might be synthetic, and wants
a second opinion before acting on it. They are not audio or ML experts. They
arrive with one clip and one question: should I trust this recording?

The project is also a university research project, assessed by markers and a
supervisor, but they are not the audience the product is designed for.

## Product Purpose

SoundSentinal scores an uploaded clip for how likely its speech is to be
synthetic. It reports that score against the model's own decision threshold
and next to the model's measured error rates.

**Success means the visitor understands the reading** (confirmed): they leave
knowing how much weight it can bear. In particular they know that a low score
is not proof a recording is real, especially for clean, studio-quality audio.
A visitor who comes away with a false "it's real" or "it's fake" is a failure,
even if the score was right.

## Positioning

**An instrument, not a verdict machine.** Other detectors stamp REAL or FAKE.
SoundSentinal shows:

- the score on a scale;
- the threshold, drawn on screen;
- the band where genuine speech sometimes reaches;
- how often the live model is wrong, measured on recordings it never trained
  on, with false accusations and misses counted separately.

Every number shown belongs to the checkpoint actually being served. A
competitor could only copy this by publishing its own calibrated error rates.

## Operating Context

- **Input:**
  - WAV, MP3 or FLAC, up to 5 MB, sent as-is.
  - MP4, M4A, MOV or WebM, up to 200 MB. The browser decodes these, mixes
    them to mono at 16 kHz and cuts them to 120 s. Only that WAV is sent.
- The model reads one 4-second window.
- **Privacy:** clips are processed in memory and never stored or logged.
- **Hosting:** the model server runs on Azure Container Apps and scales to
  zero. The first reading after a quiet spell waits a minute or more while it
  starts, and the interface says so.
- **Scale:** `/result` shows the score in log-odds. The headline is a
  confidence percentage: the model's certainty, not a measured accuracy.

## Capabilities and Constraints

- **Model:** SSL-AASIST (an XLS-R 300M front-end with an AASIST back-end). It
  reads the raw waveform, not a spectrogram. It was trained on ASVspoof2019 LA,
  SpeechFake and our own fakes from 8 open TTS families.
- **Threshold:** the live threshold is +7.07 log-odds, calibrated on
  held-out People's Speech. It is the last threshold for these weights.
- **Override:** the model is served by the owner's override of a pre-registered
  bar (In-the-Wild real flagged 2.18% against 2%). Any write-up must say so.
- **Known weaknesses.** These must stay visible in the product:
  - Clean studio fakes are missed more often: 13.7–20.7% get past.
  - Some recent LLM-codec TTS mostly passes (Qwen3-TTS, VoxCPM).
  - Clean real speech is flagged more often than noisy real speech.
  - The model only reads the first 4 seconds.
  - It is not speaker verification.
- **Numbers:** every number comes from `RESULTS.md` and the measured JSONs
  beside the checkpoint. Never type one into a page by hand.
- **Data licences:** training data must allow commercial use. No NC or ND
  licences, and no gated or sign-up datasets.
- **Future (confirmed):** it could become a product after the university
  write-up, so commercial options stay open. Hosting choices should keep that
  in mind; for example, Vercel Hobby does not allow commercial use.

## Brand Commitments

- The name is **SoundSentinal**, spelled with an "a".
- The thesis is "an instrument, not a verdict machine". `DESIGN.md` draws its
  three rules from it.
- The standing disclaimer: a reading is a signal worth following up, never
  proof that a recording is real or fake.
- Never show the raw P(spoof). It is 0.999 at the threshold and misleads.

## Evidence on Hand

- `RESULTS.md`: every measured number, with its job ID.
- `ai_model/checkpoints/best.measured-*.json`: the error rates the app shows.
- `ai_model/LA_T_*.flac`: two sample clips, one real and one fake.

There are no users, testimonials, customers, press or usage figures. Do not
invent any.

## Product Principles

1. **Calibrated honesty over reassurance.** Show the uncertainty, the
   threshold and the error rates, even when a cleaner answer would feel better.
2. **The weaknesses are part of the product.** What the model cannot do is
   stated where the reading is, not hidden in documentation.
3. **Only live numbers.** Nothing is claimed for a model or operating point
   that isn't being served.
4. **The clip stays private.** Process it in memory, store nothing, and send
   the server only what it needs.
5. **Written for a non-expert under doubt.** Plain language first; the
   technical detail is there for whoever wants to check it.

## Accessibility & Inclusion

- **Standard:** WCAG 2.2 AA.
- **Colour:** the verdict colours must survive colour-blindness. They were
  checked under protanopia, which is why teal is never a verdict colour.
  Colour is never the only channel.
- **Motion:** reduced motion is honoured, and the footer's pause switch stops
  all ambient motion for everyone.
