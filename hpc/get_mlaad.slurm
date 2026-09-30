#!/bin/bash
# Download MLAAD's English fakes (143 TTS systems, ~1,000 clips each, ~44 GB)
# at the revision mlaad.py pins, for --extra-train speechfake+mlaad and the
# held-out evaluation splits (RESULTS.md Finding 17).
#
#   sbatch hpc/get_mlaad.slurm
#
# GATED, and licensed CC BY-NC 4.0 (non-commercial). Before the first run:
#   1. accept the terms at https://huggingface.co/datasets/mueller91/MLAAD
#      while logged in to Hugging Face;
#   2. on M3, once:  source hpc/env.sh && $VENV_DIR/bin/hf auth login
#      (a read token; it is stored under $HF_HOME, not in this repo).
#
# 143k small files, so the download is resumable rather than fast: files
# already present are skipped, and the loop below retries after Hugging
# Face's rate limiter or a dropped connection. Re-submit if it hits its limit.
# Afterwards `mlaad.py check` confirms every pinned system arrived, and
# `librispeech.py check` that the real side is still held out.
#
#SBATCH --job-name=mlaad-download
#SBATCH --account=df37
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh
mkdir -p "$MLAAD_ROOT"

cd ai_model
for attempt in $(seq 1 30); do
  if "$VENV_DIR/bin/python" - <<'PY'
import os, mlaad
from huggingface_hub import snapshot_download
snapshot_download(mlaad.REPO, repo_type="dataset", revision=mlaad.REVISION,
                  local_dir=os.environ["MLAAD_ROOT"], max_workers=8,
                  allow_patterns=[f"fake/{mlaad.LANGUAGE}/**", "LICENSE",
                                  "NOTICE_NO_COMMERCIAL_USE.txt", "README.md"])
PY
  then break; fi
  echo "== attempt $attempt failed; retrying in 5 min =="
  sleep 300
done

"$VENV_DIR/bin/python" mlaad.py check "$MLAAD_ROOT"
"$VENV_DIR/bin/python" librispeech.py check "$LIBRISPEECH_ROOT"
du -sh "$MLAAD_ROOT"
