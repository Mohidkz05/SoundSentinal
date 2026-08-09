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
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas tqdm opacus flask soundfile requests
```

Use the CPU wheels unless you intend to train — only training benefits from a
GPU. For CUDA, swap the index URL for `https://download.pytorch.org/whl/cu128`.

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

First, lets install the Dependancies: ```npm install```

Now lets, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
