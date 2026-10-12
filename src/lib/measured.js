/**
 * The served model's measured error rates, as bundled with the site.
 *
 * The home page used to fetch these and show an empty panel until the model
 * server answered — up to a minute when it had scaled to zero, so most
 * visitors never saw the table at all. These are the same rows /api/model
 * returns (app.py reads them from the checkpoint's best.measured-*.json), so
 * the table renders at once and the fetch only replaces them if the live
 * server reports something different.
 *
 * Copied from /api/model for the Finding 18 checkpoint (threshold +7.07),
 * 10 October 2026. When a new checkpoint is served, refresh this file:
 * curl the route and paste `measured` (and threshold_score) here.
 */
export const MEASURED_THRESHOLD = 7.073152542114258;

export const MEASURED = [
  {
    "about": "Real-world recordings of public figures: interviews, speeches, broadcasts, and deepfakes of the same people",
    "fakes_passed": 0.018872714962762355,
    "n_fake": 11816,
    "n_real": 19963,
    "real_flagged": 0.021840404748785253,
    "set": "In-the-Wild"
  },
  {
    "about": "Clean studio speech; the fakes are 2019-era text-to-speech and voice conversion",
    "fakes_passed": 0.20653705269089884,
    "n_fake": 63882,
    "n_real": 7355,
    "real_flagged": 0,
    "set": "ASVspoof2019 LA eval"
  },
  {
    "about": "Clean modern text-to-speech and voice cloning from 26 systems — all of which the model saw in training",
    "fakes_passed": 0.13654429811828667,
    "n_fake": 189455,
    "n_real": 19200,
    "real_flagged": 0,
    "set": "SpeechFake test (English)"
  },
  {
    "about": "Clean fakes from four open text-to-speech families the model never trained on",
    "fakes_passed": 0.15271182204448888,
    "n_fake": 4001,
    "n_real": 1339,
    "real_flagged": 0.0351008215085885,
    "set": "Unseen open TTS (held-out families)"
  }
];
