# SoundSentinal

Deepfake audio detector: a [Next.js](https://nextjs.org) frontend and a Flask +
PyTorch backend that classifies an uploaded clip as real or spoofed. The model is
trained with differential privacy (Opacus) on the ASVspoof2019 corpus.

## Getting Started

### AI model (Python)

Create a virtual environment in the repo root:

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
```

If that fails on Ubuntu/WSL with "ensurepip is not available", either
`sudo apt install python3.12-venv`, or bootstrap pip without sudo:

```bash
python3 -m venv --without-pip venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python -
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

That pulls the CPU build of torch, which is what you want unless you intend to
train — only training benefits from a GPU, and the CPU wheels are ~1.2 GB
against several GB for CUDA. For CUDA, reinstall torch from the CUDA index:
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.

Check the install, then run the API:

```bash
cd ai_model
python verify_setup.py            # shapes + train/serve parity
python app.py                     # http://127.0.0.1:5000
python test_api.py                # in a second shell
```

Training needs the ASVspoof2019 dataset in `data/` (or set `$ASVSPOOF_ROOT`):

```bash
python train_dp_avspoof.py --corpus LA
```

Note: there are no trained weights in the repo, so `app.py` will refuse to start
until you train. See `CLAUDE.md` for project state and architecture details.

### Frontend

```bash
npm install
npm run dev                       # http://localhost:3000
```

### Running both halves together

The frontend talks to the model through a Next.js route at `/api/predict`,
which forwards the upload to Flask. Nothing in the browser ever addresses port
5000 directly — that avoids CORS entirely and keeps the model server off the
public surface. So an analysis needs both processes running:

```bash
cd ai_model && python app.py      # terminal 1 — http://127.0.0.1:5000
npm run dev                       # terminal 2 — http://localhost:3000
```

With Flask down, uploads fail with a message saying so rather than hanging. Set
`MODEL_API_URL` if the model server is not on `http://127.0.0.1:5000`.

## Where things are

| Path | What it is |
| --- | --- |
| `ai_model/model.py` | The network and the preprocessing. Shared by trainer and server — don't redefine either anywhere else. |
| `ai_model/train_dp_avspoof.py` | Training loop, DP via Opacus, dev-set EER, checkpointing. |
| `ai_model/app.py` | The inference server. `POST /predict` returns `spoof_probability`, `prediction` and `threshold`. |
| `src/app/` | Next.js App Router pages: `/`, `/upload`, `/result`, `/design`. |
| `src/app/globals.css` | The design system. Tokens are defined here and nowhere else. |
| `components/three/` | The ambient WebGL layer. Import scenes from `lazy.js`. |

## Project documentation

- **`CLAUDE.md`** — project state, architecture details, known gaps, roadmap.
- **`APPROACH.md`** — the model and research plan: why AASIST, and where
  differential privacy does and doesn't belong.
- **`DESIGN.md`** — the design system and the reasoning behind it. `/design`
  renders the living reference from the same CSS the product uses.
- **`HANDOFF.md`** — notes from the frontend redesign, including the WebGL
  uniform bug that is easy to reintroduce.
