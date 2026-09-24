# SoundSentinal

Deepfake audio detector. A Next.js frontend and a Flask + PyTorch backend that
classifies an uploaded audio clip as real or spoofed. The model is trained with
differential privacy (Opacus) on the ASVspoof2019 corpus.

University project. `main` is the only branch — see "Branching" below.

> **There is an uncommitted frontend redesign in the working tree.** Read
> **`HANDOFF.md`** first: it covers the full-bleed layout, the graduated
> monochrome verdict scale, the new `/upload` 3D intake surface, the rebuilt
> `/result`, and a silent WebGL uniform bug that had frozen the entire 3D layer.

## Layout

```
RESULTS.md               Every measured number and what follows from it.
ai_model/
  model.py               Models + preprocessing. SHARED by trainer and server.
                         build_transform(frontend) picks log-Mel, LFCC or raw;
                         build_model(arch) picks the CNN or AASIST. Both
                         choices are stored in the checkpoint and read back.
  aasist.py              AASIST, ported from the official implementation with
                         BatchNorm swapped for GroupNorm. Its header lists nine
                         deviations from upstream; read them before editing.
  rawboost.py            RawBoost waveform augmentation, TRAINING ONLY
                         (`--rawboost N`). Never imported by app.py or
                         evaluate.py. Bit-identical to upstream.
  evaluate.py            Scores a checkpoint on the eval partition: EER,
                         min t-DCF, per-attack breakdown, JSON out.
                         `--dataset itw` scores In-the-Wild instead.
  in_the_wild.py         Adapter for In-the-Wild (Müller et al.) — the
                         generalisation test set. EVALUATION ONLY; read its
                         header before using it for anything else.
  tdcf.py                min t-DCF, ASVspoof2019's primary metric.
  summarise_results.py   Tabulates evaluate.py's JSON files.
  train_dp_avspoof.py    DP training loop (Opacus), dev-set eval, checkpointing.
  app.py                 Flask server, POST /predict.
  verify_setup.py        Smoke test for the model/serving contract.
  test_api.py            Sends a sample .flac to a running server.
  LA_T_*.flac            Two sample clips (one bonafide, one spoof).
  train_file.txt         Two-line protocol snippet matching those clips.
requirements.txt         Pinned Python dependencies, verified versions.
src/app/                 Next.js App Router pages: /, /upload, /result, /design.
  globals.css            The design system. Single source of truth for tokens.
  api/predict/route.js   The proxy to Flask. The browser's only route to the
                         model; nothing in src/ addresses port 5000 directly.
src/lib/                 motion.js (Motion variants), verdict.js (tiers),
                         peaks.js (browser-side audio decode + envelope),
                         clip.js (carries the measured clip /upload → /result).
hpc/                     Training on Monash M3 (Slurm). README.md is the
                         runbook; env.sh holds every path. Not a VM — see below.
components/              header.js, theme.js.
  ui/                    button.js, calibration-meter.js, verdict-scale.js,
                         mark.js.
  three/                 The WebGL layer. lazy.js is the entry point —
                         import scenes from there, never directly.
```

## Design system

See **`DESIGN.md`** for the rules and the reasoning, and **`/design`** for the
living reference — it renders from the same CSS the product does, so it can't
drift. Tokens are defined once in `src/app/globals.css`; don't write one-off
colours, radii or durations in components.

Thesis: **an instrument, not a verdict machine.** The model returns a
probability, so the UI reports a reading against a visible threshold rather than
stamping a verdict. Three rules follow: the brand colour never renders a result;
the result scale desaturates where the model is least certain; the decision
threshold is drawn on screen.

Two things that follow from the thesis and are easy to undo by accident:

- **A reading is never drawn as a filled bar.** The diverging ramp
  (`verdict-ramp`) is only rendered raw on `/design`, as palette documentation.
  Readings go through `components/ui/verdict-scale.js`, which masks the ramp
  into an engraved graduated scale — a continuous fill reads as a progress bar,
  i.e. a quantity accumulating towards completion, which is not what a
  probability is. Shared by the meter, the home illustration and `/design`;
  don't hand-roll a second copy.
- **One measure for the whole app**: the `shell` utility (`--shell`, 84rem),
  used by the header and every page. Widening it must never widen a paragraph —
  prose keeps its own `ch` measure, and what earns the width is the scale and
  the waveform, where width is resolution. See "Layout" in `DESIGN.md`.

Brand is Sentinel Teal (hue 190.3°, viridis at 0.50). The verdict scale is a
diverging violet↔orange (the ends of `plasma`, same axis as ColorBrewer PuOr).
Those are separate on purpose — teal↔ember was measured at ΔE 0.048 under
protanopia, i.e. indistinguishable, so the brand colour *cannot* double as the
verdict colour. Type is Archivo (two widths off the variable width axis) with
IBM Plex Mono for every number.

Note `motion` (Framer Motion), `three` and `@react-three/fiber` are now
dependencies, and `npm install` has been run — `node_modules/` exists.

**The 3D layer**: six WebGL scenes in `components/three/`, each depicting
something the model actually does — the log-Mel spectrogram it reads, the
waveform of the clip you uploaded, the uncertainty around its threshold. All of
it is ambient: every canvas is `aria-hidden`, carries no meaning the DOM doesn't
already carry, and is absent entirely without WebGL. See **"The 3D layer"** in
`DESIGN.md` for the six rules, and `/design` for the living reference.

Three things to know before touching it:

- **Reach uniforms through a ref on the material**, never through the object the
  component built with `useMemo` and passed as a `uniforms` prop. Mutating that
  object in `useFrame` is a **silent no-op** — the writes land somewhere the GPU
  never reads, nothing errors, and the scene renders frozen at its initial
  values, which looks exactly like a tuning problem. All five shader scenes
  shipped with this bug and were static until it was found in August 2026. To
  test whether a per-frame write is landing, set `uOpacity` to 0 every frame and
  look: if the scene is still visible, the write is going nowhere. Details in
  `HANDOFF.md`.

- **Import scenes from `components/three/lazy.js`, not from the scene files.**
  three.js is ~250 kB. The lazy module wraps every scene in
  `dynamic(..., { ssr: false })`, which is what keeps First Load JS at ~108 kB
  rather than ~350 kB. Importing a scene directly pulls three.js into the
  initial bundle and silently triples it.
- **Shaders read colour from the design tokens** via `usePalette()`, never from
  hex literals, so both themes and any future palette change reach the geometry.
- **No page wrapper may set an opaque background.** The ambient backdrop in
  `layout.js` sits at a negative z-index — above the body's canvas colour,
  below every panel. A `bg-canvas` on a page's outer `div` (which is what these
  pages used to have) paints straight over it.

## Setup

A working venv already exists at `venv/` (CPU torch 2.13.0, torchaudio 2.11.0).
Activate it with `source venv/bin/activate`.

To rebuild it from scratch on this machine, note that Ubuntu splits `ensurepip`
into a separate `python3.12-venv` package which is **not installed here**, and
installing it needs sudo in an interactive terminal. The sudo-free workaround:

```bash
python3 -m venv --without-pip venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python -
./venv/bin/pip install -r requirements.txt
```

CPU wheels are deliberate: ~1.2 GB installed vs several GB for CUDA, and nothing
except training needs a GPU. There *is* a working RTX 4070 visible from WSL
(`/dev/dxg` present, driver 610.62), so if you start training here rather than on
Windows, swap in the CUDA build with
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129`.
Note **cu129, not cu128** — the cu128 index stops at torch 2.11.0, so the pinned
2.13.0 will not resolve there. `hpc/requirements-cuda.txt` carries the same pins
for the cluster and explains the fallback.

**Training properly happens on Monash M3**, not here — project `df37`, granted
14 September 2026. `hpc/README.md` is the runbook and opens with why "it is not
a VM" changes the workflow. The short version is
`bash hpc/bootstrap.sh` → `sbatch hpc/get_la.slurm` → `sbatch hpc/train.slurm
--no-dp`. Two env vars carry the layout into the Python: `ASVSPOOF_ROOT` for the
corpus and `CKPT_ROOT` for the checkpoint tree, the latter read by
`train_dp_avspoof.py`, `app.py` and `verify_setup.py` alike. Unset, all three
fall back to the in-repo paths and this machine behaves exactly as before.

```bash
# Frontend (node v24 via nvm; node_modules is not installed yet)
npm install
npm run dev                       # http://localhost:3000
```

Both halves must run for an analysis to work — the page posts to `/api/predict`,
which forwards to Flask:

```bash
cd ai_model
../venv/bin/python verify_setup.py       # shapes + train/serve parity
../venv/bin/python train_dp_avspoof.py --corpus LA
../venv/bin/python train_dp_avspoof.py --corpus LA --no-dp   # non-private baseline
../venv/bin/python train_dp_avspoof.py --arch aasist --no-dp # AASIST (slow on CPU)
../venv/bin/python evaluate.py --arch aasist                 # score it on eval
../venv/bin/python app.py                # http://127.0.0.1:5000
../venv/bin/python test_api.py           # in a second shell
```

## The one rule: model.py is the single source of truth

`train_dp_avspoof.py`, `evaluate.py` and `app.py` all import `build_model()`,
`build_transform()`, and `preprocess_waveform()` from `ai_model/model.py`.
`aasist.py` defines a network and nothing else — it is imported *by* model.py,
never around it, so there is still exactly one place that decides what a
checkpoint means.

**Never construct a network directly.** `build_model(arch)` exists because a
checkpoint records which architecture produced it, and the loader has to honour
that rather than build whatever is currently the default. The same goes for
`build_transform(frontend)`. Pairing the two wrongly — AASIST on a spectrogram,
the CNN on a waveform — is caught by `check_pairing()`, because it would not
crash: it would train on nonsense and report a number.

This is not stylistic. Those two files each used to carry their own copy of the
architecture and the preprocessing, and they silently diverged — the trainer
moved to log-Mel + per-sample standardization while the server stayed on raw
power Mel, so inference fed the model a different distribution than it trained
on. Nothing crashed; predictions were just quietly wrong.

**Do not redefine the network or the preprocessing steps anywhere else.** If you
change `model.py`, run `python verify_setup.py` — it asserts that the tensor
`app.py` produces is byte-identical to the one the training dataset produces.

## Architecture and data details

- **Audio loading**: `load_audio()` in `model.py` uses **soundfile**, not
  `torchaudio.load`. As of torchaudio 2.11 that call delegates to TorchCodec,
  which requires system FFmpeg libraries (absent here, and installing them needs
  sudo). soundfile bundles libsndfile in the wheel, so FLAC works with no system
  packages. Don't switch back to `torchaudio.load`.
- **Input**: mono, 16 kHz, padded/truncated to 64000 samples (4 s) → log-Mel
  spectrogram (`n_fft=1024`, `hop=512`, `n_mels=128`, `top_db=80`) → per-sample
  standardization. Shape `(1, 128, 126)`.
- **Networks**: two, chosen with `--arch`.
  - `cnn` (default, 267k params) — 2× (Conv2d → ReLU → MaxPool) →
    `AdaptiveAvgPool2d((8,8))` → `Linear(2048, 128)` → dropout → `Linear(128, 2)`.
    The adaptive pool is what makes the model tolerate different spectrogram
    widths; an earlier version hardcoded a flattened size of 31744 derived from
    a dummy forward pass.
  - `aasist` (297k params) and `aasist-l` (85k) — raw waveform through a
    learnable sinc filterbank, six residual blocks, then a spectro-temporal
    graph attention network. Input is `(batch, 1, 64000)`, not a spectrogram.
    See `ai_model/aasist.py`.
- **Per-architecture training recipe**: `TRAIN_DEFAULTS` in
  `train_dp_avspoof.py`. The CNN trains 5 epochs at batch 64, lr 1e-3; AASIST
  100 epochs at batch 24, lr 1e-4 with weight decay and cosine annealing, which
  is its published recipe. Neither number is a free choice for AASIST — at
  batch 64 its first residual block alone holds a 4.2 GB activation. **So the
  AASIST row differs from the CNN rows by schedule as well as architecture**,
  and any writeup comparing them has to say so.
- **Labels**: `bonafide=0`, `spoof=1`. The API reports these as
  `"Real Audio"` / `"Deepfake Audio"` (`CLASS_NAMES` in `model.py`).
- **DP**: `noise_multiplier=1.1`, `max_grad_norm=1.0`, `delta=1e-5`. Epsilon is
  printed each epoch. Opacus wraps the model, so weights are saved from
  `model._module.state_dict()`.
- **Dataset**: not in the repo. Expected at `data/LA/...` or wherever
  `$ASVSPOOF_ROOT` points. `data/` and `checkpoints/` are gitignored.

## Checkpoints

`train_dp_avspoof.py` writes to
`ai_model/checkpoints/<arch>/[<frontend>/][nodp/]`, where the default
architecture and front-end are omitted — so the CNN log-Mel DP run is still
plain `checkpoints/` and its baseline still `checkpoints/nodp/`, exactly as
before, while AASIST lands in `checkpoints/aasist/nodp/`. Each directory gets a
timestamped file per epoch, plus rolling `last.pth` (auto-resume) and
`best.pth` (lowest dev EER). Separate directories are what stop one run
auto-resuming from another's weights; a mismatch is caught and named on resume.

`evaluate.py --arch aasist` resolves that path for you, so the directory layout
only has to be typed when using `--ckpt` for something unusual.

**Every `torch.load` of a checkpoint must pass `weights_only=False`.** torch 2.6
changed that default to `True`, which refuses any checkpoint containing a
non-tensor object — and ours do: Opacus's `get_epsilon()` returns a numpy
scalar, so every DP checkpoint carries one inside `metrics`. `app.py` failed
this way at import time on M3 while passing locally, because locally there is no
real checkpoint and `verify_setup.py`'s stand-in held only plain tensors. That
stand-in now carries a numpy scalar on purpose — a fixture easier to load than
the real thing tests nothing. New checkpoints also store `float(epsilon)`, so
they do not depend on the flag.
`app.py` loads `checkpoints/best.pth`, falling back to a legacy flat
`deepfake_audio_detector.pth` if present.

**There are currently no valid trained weights in the repo.** A 16 MB
`deepfake_audio_detector.pth` used to be committed, but it was trained against
the old hardcoded-size architecture and cannot load into the current model. It
was untracked (the file may still be on disk locally, and is now covered by the
`*.pth` ignore rule). Training must be re-run to produce usable weights.

## Known gaps — the real state of things

Be honest about these rather than assuming they work:

1. **Models are trained and measured — see `RESULTS.md`.** Four runs exist on
   M3 (CNN log-Mel non-private, CNN LFCC non-private, CNN DP at ε=0.48, and
   AASIST non-private), scored on the eval partition with EER and min t-DCF.
   Best result is **AASIST at 3.17% EER / 0.0909 min t-DCF** (`best.pth`,
   epoch 42). That beats every CNN run (best 9.60% / 0.2124) and both official
   GMM baselines, but is about 3.3× off the paper's 0.83%, probably partly
   because of the GroupNorm swap. **AASIST's `best.pth` was copied to
   `ai_model/checkpoints/best.pth` on 23 September 2026**, so `app.py` serves
   it locally (gitignored — a fresh clone has no weights; `scp` it from M3's
   `checkpoints/aasist/nodp/`). Use host `m3`, not `m3-dtn`: the latter's host
   key isn't in `known_hosts` on this laptop. The corpus lives on M3, not here —
   `data/` is still absent and `$ASVSPOOF_ROOT` unset locally.

   **In-the-Wild is scored (21 September 2026): the models collapse.** 31,779
   clips at `$ITW_ROOT` on M3. AASIST goes from 3.17% to **37.15% EER**; the
   log-Mel CNN scores **58.54%**, i.e. worse than chance. At the dev-calibrated
   thresholds both models flag most *real* clips as fake (AASIST 73%, CNN
   97%), so the product as it stands would mislabel most genuine modern audio.
   Finding 6 in `RESULTS.md`. It is an **evaluation set only** — never train,
   select or calibrate on it, or every LA row in `APPROACH.md` stops being
   comparable to published work and the CC-BY-SA licence reaches a model
   artifact.

   **Two fixes tried on 24 September 2026 did not help In-the-Wild** (Finding 7):
   RawBoost gives the best LA result yet (1.74% EER / 0.0531 min t-DCF) but
   48.78% on In-the-Wild; adding ASVspoof 5 train (`--extra-train asvspoof5`,
   branch `asvspoof5`) scores 38.14%. The served `best.pth` is still the
   unaugmented AASIST. The next candidate is an SSL front-end.

   **AASIST is trained (20–21 September 2026)**, non-private only. There is no
   DP AASIST run yet, so the cost of privacy has only been measured on the CNN.
2. **Model selection is known-broken.** `save_ckpt` picks `best.pth` by dev
   EER, and dev reuses the training attacks; measured, it selects a worse model
   than an earlier epoch. Finding 1 in `RESULTS.md`. AASIST suffers less —
   its `best.pth` is 0.19 points of eval EER off the best epoch (Finding 5) —
   but ranks only 23rd of 100. The same flaw affects the calibrated
   threshold. Both need a held-out set with unseen attacks.

3. **No tests** beyond `verify_setup.py` (a shape/parity smoke test) and
   `test_api.py` (a manual one-shot client). No CI. In particular there is no
   automated check that the 3D scenes still animate — the uniform-ref bug was
   invisible for months and would be again. The check that catches it is
   cheap: screenshot a canvas region twice a few seconds apart and diff them; a
   frozen scene reads exactly zero.
4. **`app.py` is still the Werkzeug development server.** `debug=True` is gone
   (it exposed an interactive debugger that executes code) and it binds to
   loopback, but use `waitress` or `gunicorn` before this is hosted anywhere.
5. **The upload limit is enforced in three places** — the page, the proxy route
   and Flask's `MAX_CONTENT_LENGTH` — and all three say 5 MB. Changing one means
   changing all three; they are cross-referenced by comment.

## Roadmap

> **See `APPROACH.md` for the model and research plan** — decided 17 August 2026.
> It supersedes parts of this section: the chosen architecture is **AASIST**, DP
> is **off for the main results** (but every architecture stays DP-compatible —
> no BatchNorm), and the comparison table with verified EERs and their sources
> lives there. The engineering notes below are still current; where the two
> disagree about *strategy*, `APPROACH.md` wins.

Steps 1 and 2 below are **implemented but never run** (August 2026); everything
after them is analysis, recorded so it isn't re-derived. Suggested order: non-DP
baseline + class weights → retrain → threshold plumbed end to end → API returns
`spoof_probability` → proxy route + upload wiring → then architecture work. That
produces a working vertical slice with honest numbers early, so later changes can
be measured against something.

### Model accuracy, in priority order

1. **Get a non-private baseline first.** *Code done, not yet trained.*
   `--no-dp` skips `make_private` and `get_epsilon`; `unwrap()` handles the model
   being bare rather than Opacus-wrapped. Without a baseline you cannot tell
   whether a bad result comes from the architecture, the data pipeline, or DP
   noise — three different fixes. The baseline is the ceiling; the gap to the DP
   run is the measured cost of privacy, which is also the interesting result to
   report.

   **DP and baseline runs write to separate checkpoint directories** —
   `checkpoints/` for DP, `checkpoints/nodp/` for the baseline. They share an
   architecture but not a training regime, so a shared `last.pth` would let one
   run silently auto-resume from the other's weights. `app.py` still reads
   `checkpoints/best.pth`, i.e. the DP run.

2. **Fix the class imbalance.** *Code done, not yet trained.* ASVspoof2019 LA
   train is roughly 2,580 bonafide vs 22,800 spoof (~1:9). Unweighted
   `CrossEntropyLoss` drifts toward predicting "spoof" for everything while still
   looking accurate. `compute_class_weights()` derives inverse-frequency weights
   from the actual protocol counts rather than hardcoding them (on LA train that
   comes out to `[4.92, 0.56]`, a ratio of 8.84), prints them, and stores them in
   the checkpoint. `--no-class-weights` turns it off for ablation.

   **Do not use `WeightedRandomSampler`** — Opacus's `make_private` replaces the
   DataLoader's sampler with Poisson sampling for privacy accounting, so a custom
   sampler is silently discarded. Weight the loss instead.

3. **Use the EER threshold that is already computed.** *Done.* `save_ckpt`
   stores `threshold` (and the full dev metrics, including a confusion matrix)
   in every checkpoint; `app.py` loads it, applies it instead of
   `torch.max(outputs)`, and reports it in the response along with
   `threshold_calibrated`, which is false when the checkpoint carries none and
   0.5 is being used. `/result` says so on screen in that case.

4. **Report eval-set EER, not dev.** The dev partition uses attacks A01–A06, the
   same ones seen in training. The eval partition has unseen A07–A19. Dev EER is
   optimistic; quote eval in any writeup.

5. **Then increase capacity.** *Superseded by the AASIST port — see
   `APPROACH.md`.* Growing the CNN to 4–6 blocks is no longer the plan, because
   `--arch aasist` is a better-evidenced 297k-parameter model that is already
   wired in. Keep the rule that motivated this step: **do not add BatchNorm**,
   anywhere, in any architecture. It mixes statistics across a batch, breaking
   DP-SGD's per-sample gradient guarantee, and Opacus rejects such models. Use
   `GroupNorm`. `verify_setup.py` asserts this for every architecture and also
   takes one real DP-SGD step with each, because "no BatchNorm" turned out to
   be necessary and not sufficient — see the AASIST notes in `APPROACH.md`.

6. **Cheap wins after that:** random 4-second crops instead of always truncating
   from the start; SpecAugment (`torchaudio.transforms.TimeMasking` /
   `FrequencyMasking`); more epochs — though each epoch spends privacy budget.

Also worth logging a confusion matrix, not just accuracy and EER. On 1:9 data,
"90% accurate" can mean "always guesses spoof".

### Bridging frontend and backend — done, and how it fits together

Wired 21 August 2026 and verified in a real browser end to end. The shape of it,
because each piece was chosen for a reason worth not undoing:

**The response is a probability, not a verdict.** `POST /predict` returns
`spoof_probability` (class 1 = spoof), `prediction`, `threshold`,
`threshold_calibrated`, and a `model` block read off the checkpoint. It used to
return a label plus the confidence of *whichever class won* — `Real Audio, 90%`
means P(spoof) = 10%, and the four graded tiers on `/result` cannot be recovered
from that without unpacking the label. A continuous probability is the only
thing a graduated scale can be drawn from.

**The browser never addresses Flask.** `src/app/api/predict/route.js` forwards
the upload to `$MODEL_API_URL` (default `http://127.0.0.1:5000`). No CORS, no
`flask-cors`, the model port stays off the public surface, and one env var moves
at deploy time. The route also normalises every failure to `{ error }` with a
sensible status, so the page has one shape to handle: 400 no file, 413 too
large, 422 undecodable, 502 upstream nonsense, 503 model server unreachable
(with the command to start it).

**The clip and the reading travel together.** `src/lib/clip.js` stores both in
`sessionStorage` under one key. `/upload` posts first and navigates second, so
`/result` never renders before its number exists. Arriving at `/result` directly
shows a waiting state — no fabricated reading, no model card.

**The model card is built from the API response**, never written down in the
page. A hand-maintained card describes whichever run someone last remembered to
type in, which looks like provenance while being fiction.

**Two rules that are easy to break by accident:**

- **No tier description may mention which side of the threshold it falls on.**
  The tiers are fixed quarters of the range; the threshold moves with every
  training run. Two tier strings used to say "sits below the decision
  threshold" / "clears the threshold", which was true only while the threshold
  was pinned at 0.5. The first calibrated checkpoint served 0.413 and the page
  contradicted itself on screen. The threshold relation is stated once, by the
  page, from the value the API returned.
- **The 5 MB limit lives in three places** — `MAX_BYTES` in `upload/page.js`,
  `MAX_BYTES` in the proxy route, `MAX_UPLOAD_BYTES` in `app.py`. The client
  check is a courtesy; the route is reachable directly and Flask has no default
  cap, so all three are real. `libsndfile` does decode MP3, but that path is
  untested — no MP3 sample exists in the repo.

### Stack review — what to change and what to leave alone

**Question DP-SGD's premise, not Opacus.** Opacus 1.6.0 is the right library and
already defaults to the `prv` accountant (the tighter epsilon bound — older
guides tell you to opt in; this repo is already there). There is no better
PyTorch DP-SGD option. The real issue is what DP-SGD buys here: **it protects the
privacy of the training data — ASVspoof2019, a publicly released corpus.** It
costs accuracy to prevent memorizing speakers whose audio is already published.
Meanwhile the privacy concern a user of this tool actually has is about the clip
they upload, which DP-SGD does nothing for (that needs no logging, in-memory
processing, immediate deletion). Two honest paths: keep DP and frame the project
as measuring its cost (needs the baseline above), or drop it for the main model
and state inference-time privacy properties instead. **Resolved 17 Aug 2026 —
both: DP off for the main results, architectures kept DP-compatible, and the
cost-of-privacy gap measured as the research contribution. See `APPROACH.md`.**
DP-SGD becomes genuinely
justified the moment training uses user-contributed voice data.

**Replace: Mel → LFCC features.** Cheapest real accuracy win, a one-line change
in `build_transform()`; `torchaudio.transforms.LFCC` is already available in the
installed version. The Mel scale is designed to mimic human hearing and so
deliberately compresses high frequencies — exactly where vocoder and synthesis
artifacts live. This is why the official ASVspoof2019 baselines are LFCC-GMM and
CQCC-GMM. Run it against the current log-Mel with the same model.

**Replace: the dataset, for results that mean anything today.** ASVspoof2019's LA
attacks predate neural codec models and current commercial voice cloning, so a
model scoring well on LA eval can still fail on a modern TTS clip. **ASVspoof 5**
(2024) has modern attacks; **In-the-Wild** (Müller et al.) is an excellent
generalization test set. Even just *evaluating* the LA-trained model on
In-the-Wild gives a far more honest number, and the gap is itself a finding.

**Add: a Python dependency manifest.** *Partly done* — `requirements.txt` now
pins every dependency at the verified version and carries the CPU torch index,
so `pip install -r requirements.txt` is the whole install. Still worth **`uv`**, which
bundles its own Python/venv handling and would have entirely avoided the
`ensurepip` problem documented in Setup (no `python3.12-venv`, no sudo, no
`get-pip.py`). Highest practical-value item for a repo cloned onto several
machines.

**Add: experiment tracking.** DP vs non-DP, Mel vs LFCC, threshold sweeps — and
metrics currently go to stdout and vanish. `torch.utils.tensorboard` needs no
extra infrastructure and ships with PyTorch. Weights & Biases if a shareable
dashboard is wanted for a group project.

**Keep — don't churn these:**
- **soundfile** for I/O (see Architecture notes; don't revert to `torchaudio.load`)
- **torchaudio** for transforms — current and fine, despite its I/O migration
- **Flask** — FastAPI is nicer for ML serving, but for one endpoint it's a rewrite
  for marginal gain. The real serving problem is `debug=True` and the dev server;
  fix those and use `waitress`/`gunicorn`
- **Next.js 15.4.6 / React 19.1** — Next 16 exists; upgrading mid-project buys
  nothing here
- **npm** — pnpm is faster, not worth switching now

**On the architecture ceiling:** the 2-conv CNN is far below what's achievable —
AASIST and wav2vec2/WavLM front-ends reach ~1% EER on LA eval versus roughly
10–20% expected here. But SSL front-ends are large, interact badly with Opacus
(per-sample gradients, no BatchNorm) and would strain 8 GB of VRAM. Effort-to-
accuracy ranking: **LFCC first, then class weighting, then a deeper GroupNorm
CNN.** Reach for pretrained SSL only after dropping DP.

## Branching

`main` only. `Alex-development` (the AI model, merged via PR #1) and
`Mohid-fixes` (the training-loop rewrite) were both merged and deleted in
August 2026. Branch from `main` for new work.

Note: git identity is set repo-locally (`git config user.name` / `user.email`),
not globally, because this machine had no global git identity configured.
