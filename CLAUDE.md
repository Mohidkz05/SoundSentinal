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

```bash
# Python side, from the repo root
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas tqdm opacus flask soundfile requests

# Frontend
npm install
npm run dev                       # http://localhost:3000
```

```bash
cd ai_model
python verify_setup.py            # confirms shapes + train/serve parity
python train_dp_avspoof.py --corpus LA
python app.py                     # http://127.0.0.1:5000
python test_api.py                # in a second shell
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
2. **The model has never been evaluated end-to-end since the merge.** The
   architecture and preprocessing changed; no run has happened against them.
   Any accuracy or EER number predating that merge is meaningless now.
3. **`app.py` runs with `debug=True`** and no CORS headers. Both need attention
   before the frontend can call it from `localhost:3000`.
4. **No tests** beyond `verify_setup.py` (a shape/parity smoke test) and
   `test_api.py` (a manual one-shot client). No CI.
5. **README.md is still largely the create-next-app boilerplate**, with the
   Python setup steps bolted on top.

## Branching

`main` only. `Alex-development` (the AI model, merged via PR #1) and
`Mohid-fixes` (the training-loop rewrite) were both merged and deleted in
August 2026. Branch from `main` for new work.

Note: git identity is set repo-locally (`git config user.name` / `user.email`),
not globally, because this machine had no global git identity configured.
