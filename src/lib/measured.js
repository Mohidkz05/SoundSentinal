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
 * Copied from the server's response for the Finding 23 R0 checkpoint
 * (threshold +6.58), 13 October 2026. When a new checkpoint is served, refresh this file:
 * curl the route and paste `measured` (and threshold_score) here.
 */
export const MEASURED_THRESHOLD = 6.581302642822266;

export const MEASURED = [
  {
    "about": "Real-world recordings of public figures: interviews, speeches, broadcasts, and deepfakes of the same people",
    "fakes_passed": 0.0258124576844956,
    "n_fake": 11816,
    "n_real": 19963,
    "real_flagged": 0.006261583930271001,
    "set": "In-the-Wild"
  },
  {
    "about": "Clean studio speech; the fakes are 2019-era text-to-speech and voice conversion",
    "fakes_passed": 0.15182680567295953,
    "n_fake": 63882,
    "n_real": 7355,
    "real_flagged": 0.0004078857919782461,
    "set": "ASVspoof2019 LA eval"
  },
  {
    "about": "Clean modern text-to-speech and voice cloning from 26 systems \u2014 all of which the model saw in training",
    "fakes_passed": 0.09135678657200919,
    "n_fake": 189455,
    "n_real": 19200,
    "real_flagged": 0.0005208333333333333,
    "set": "SpeechFake test (English)"
  },
  {
    "about": "Clean fakes from four open text-to-speech families the model never trained on",
    "fakes_passed": 0.05073731567108223,
    "n_fake": 4001,
    "n_real": 1339,
    "real_flagged": 0.008215085884988798,
    "set": "Unseen open TTS (held-out families)"
  }
];
