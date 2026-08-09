# SoundSentinal

Deepfake audio detector. A Next.js frontend and a Flask + PyTorch backend that
classifies an uploaded audio clip as real or spoofed. The model is trained with
differential privacy (Opacus) on the ASVspoof2019 corpus.

University project. `main` is the only branch — see "Branching" below.

## Layout

```
ai_model/
  model.py               Model + preprocessing. SHARED by trainer and server.
  train_dp_avspoof.py    DP training loop (Opacus), dev-set eval, checkpointing.
  app.py                 Flask server, POST /predict.
  verify_setup.py        Smoke test for the model/serving contract.
  test_api.py            Sends a sample .flac to a running server.
  LA_T_*.flac            Two sample clips (one bonafide, one spoof).
  train_file.txt         Two-line protocol snippet matching those clips.
src/app/                 Next.js App Router pages: /, /upload, /result.
components/              header.js, theme.js.
```

## Setup

A working venv already exists at `venv/` (CPU torch 2.13.0, torchaudio 2.11.0).
Activate it with `source venv/bin/activate`.

To rebuild it from scratch on this machine, note that Ubuntu splits `ensurepip`
into a separate `python3.12-venv` package which is **not installed here**, and
installing it needs sudo in an interactive terminal. The sudo-free workaround:

```bash
python3 -m venv --without-pip venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python -
./venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install pandas tqdm opacus flask soundfile requests
```

CPU wheels are deliberate: ~1.2 GB installed vs several GB for CUDA, and nothing
except training needs a GPU. There *is* a working RTX 4070 visible from WSL
(`/dev/dxg` present, driver 610.62), so if you start training here rather than on
Windows, swap in the CUDA build with
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.

```bash
# Frontend (node v24 via nvm; node_modules is not installed yet)
npm install
npm run dev                       # http://localhost:3000
```

```bash
cd ai_model
../venv/bin/python verify_setup.py       # shapes + train/serve parity
../venv/bin/python train_dp_avspoof.py --corpus LA
../venv/bin/python app.py                # http://127.0.0.1:5000
../venv/bin/python test_api.py           # in a second shell
```

## The one rule: model.py is the single source of truth

`train_dp_avspoof.py` and `app.py` both import `AudioClassifierCNN`,
`build_transform()`, and `preprocess_waveform()` from `ai_model/model.py`.

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
- **Network**: 2× (Conv2d → ReLU → MaxPool) → `AdaptiveAvgPool2d((8,8))` →
  `Linear(2048, 128)` → dropout → `Linear(128, 2)`. The adaptive pool is what
  makes the model tolerate different spectrogram widths; an earlier version
  hardcoded a flattened size of 31744 derived from a dummy forward pass.
- **Labels**: `bonafide=0`, `spoof=1`. The API reports these as
  `"Real Audio"` / `"Deepfake Audio"` (`CLASS_NAMES` in `model.py`).
- **DP**: `noise_multiplier=1.1`, `max_grad_norm=1.0`, `delta=1e-5`. Epsilon is
  printed each epoch. Opacus wraps the model, so weights are saved from
  `model._module.state_dict()`.
- **Dataset**: not in the repo. Expected at `data/LA/...` or wherever
  `$ASVSPOOF_ROOT` points. `data/` and `checkpoints/` are gitignored.

## Checkpoints

`train_dp_avspoof.py` writes to `ai_model/checkpoints/`: a timestamped file each
epoch, plus rolling `last.pth` (auto-resume) and `best.pth` (lowest dev EER).
`app.py` loads `checkpoints/best.pth`, falling back to a legacy flat
`deepfake_audio_detector.pth` if present.

**There are currently no valid trained weights in the repo.** A 16 MB
`deepfake_audio_detector.pth` used to be committed, but it was trained against
the old hardcoded-size architecture and cannot load into the current model. It
was untracked (the file may still be on disk locally, and is now covered by the
`*.pth` ignore rule). Training must be re-run to produce usable weights.

## Known gaps — the real state of things

Be honest about these rather than assuming they work:

1. **The frontend is not wired to the backend.** There is no `fetch` anywhere in
   `src/` or `components/`. `/upload` and `/result` are UI shells; nothing calls
   `POST http://127.0.0.1:5000/predict`. This is the biggest missing piece.
2. **No model has been trained against the current architecture.** The plumbing
   is verified end-to-end (server loads weights, accepts a FLAC upload, returns
   JSON — confirmed with a temporary untrained checkpoint, which predicted at
   ~51%, i.e. chance, as expected). But no real training run has happened since
   the architecture and preprocessing changed. Any accuracy or EER number
   predating that change is meaningless now.
3. **`app.py` runs with `debug=True`** and no CORS headers. Both need attention
   before the frontend can call it from `localhost:3000`.
4. **No tests** beyond `verify_setup.py` (a shape/parity smoke test) and
   `test_api.py` (a manual one-shot client). No CI.
5. **README.md's lower half is still create-next-app boilerplate** (the "Learn
   More" / "Deploy on Vercel" sections). The setup instructions at the top are
   current and verified.

## Roadmap

Nothing below has been implemented — this is the analysis, recorded so it isn't
re-derived. Suggested order: non-DP baseline + class weights → retrain →
threshold plumbed end to end → API returns `spoof_probability` → proxy route +
upload wiring → then architecture work. That produces a working vertical slice
with honest numbers early, so later changes can be measured against something.

### Model accuracy, in priority order

1. **Get a non-private baseline first.** Add a `--no-dp` flag that skips
   `make_private`, train it, record dev EER. Without it you cannot tell whether a
   bad result comes from the architecture, the data pipeline, or DP noise — three
   different fixes. The baseline is the ceiling; the gap to the DP run is the
   measured cost of privacy, which is also the interesting result to report.

2. **Fix the class imbalance.** ASVspoof2019 LA train is roughly 2,580 bonafide
   vs 22,800 spoof (~1:9). Unweighted `CrossEntropyLoss` drifts toward predicting
   "spoof" for everything while still looking accurate. Use class weights derived
   from the actual protocol counts:
   `nn.CrossEntropyLoss(weight=torch.tensor([9.0, 1.0], device=DEVICE))`.

   **Do not use `WeightedRandomSampler`** — Opacus's `make_private` replaces the
   DataLoader's sampler with Poisson sampling for privacy accounting, so a custom
   sampler is silently discarded. Weight the loss instead.

3. **Use the EER threshold that is already computed.** `compute_eer_np` returns a
   calibrated threshold, `save_ckpt` discards it, and `app.py` then uses
   `torch.max(outputs)` — an implicit 0.5 cutoff. The tuned operating point never
   reaches production. Persist `thresh` in the checkpoint and apply it when
   serving; on imbalanced DP-trained models 0.5 is rarely right.

4. **Report eval-set EER, not dev.** The dev partition uses attacks A01–A06, the
   same ones seen in training. The eval partition has unseen A07–A19. Dev EER is
   optimistic; quote eval in any writeup.

5. **Then increase capacity.** Two conv layers is small; 4–6 blocks with more
   channels is the next step. **Critical: do not add BatchNorm.** It mixes
   statistics across a batch, breaking DP-SGD's per-sample gradient guarantee —
   Opacus rejects such models. Use `GroupNorm`.

6. **Cheap wins after that:** random 4-second crops instead of always truncating
   from the start; SpecAugment (`torchaudio.transforms.TimeMasking` /
   `FrequencyMasking`); more epochs — though each epoch spends privacy budget.

Also worth logging a confusion matrix, not just accuracy and EER. On 1:9 data,
"90% accurate" can mean "always guesses spoof".

### Bridging frontend and backend

**There is a contract mismatch to resolve before writing any wiring code.**
`result/page.js` has four graded tiers (unlikely → possibly → likely →
veryLikely), which needs a continuous P(spoof). But `app.py` returns a binary
label plus `confidence` of *whichever class won* — when it says
`Real Audio, 90%`, P(spoof) is 10%, and the tier cannot be recovered without
unpacking the label. Change the response to return the spoof probability
directly:

```python
probs = F.softmax(outputs, dim=1)
spoof_prob = probs[0][1].item()   # class 1 = spoof
return jsonify({
    "spoof_probability": spoof_prob,
    "prediction": CLASS_NAMES[1 if spoof_prob >= THRESHOLD else 0],
    "threshold": THRESHOLD,
})
```

Then the page maps `spoof_probability` onto its buckets, and the hardcoded
`useState('unlikely')` plus the four demo buttons come out.

- **Proxy through a Next.js route** — add `src/app/api/predict/route.js` that
  forwards the upload to `http://127.0.0.1:5000/predict`. This avoids CORS
  entirely (no `flask-cors` needed), keeps the Flask port off the public surface,
  and means one env var changes at deploy time. Calling `:5000` directly from the
  browser hits CORS immediately and forces exposing the model server.
- **`upload/page.js` has no submit path** — it stores the file in state and
  stops. Needs a submit button, a `FormData` POST, loading and error states, and
  navigation carrying the result (`sessionStorage` is fine here).
- **Enforce the advertised limits.** The page says "up to 5mb" and "MP3, Wav" but
  nothing checks either, and Flask has no default request cap. Set
  `app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024`, add `accept="audio/*"` to
  the file input, validate client-side too, and add FLAC to the listed formats
  since the sample files are FLAC. libsndfile 1.2.2 does decode MP3, but that
  path is untested — no MP3 sample exists in the repo.
- **Turn off `debug=True`** before this is reachable from a browser; it exposes
  an interactive debugger that executes code.

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
and state inference-time privacy properties instead. DP-SGD becomes genuinely
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

**Add: a Python dependency manifest.** There is none — dependencies exist only as
prose in the README. `requirements.txt` at minimum. Better: **`uv`**, which
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
