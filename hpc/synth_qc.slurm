#!/bin/bash
# Whisper intelligibility check on generated fakes (synth/qc.py, Finding 18).
#
#   sbatch hpc/synth_qc.slurm heldout-a/speecht5/speecht5 [more model dirs...]
#
#SBATCH --job-name=synth-qc
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh
export HF_HOME="$SYNTH_HF_HOME" UV_CACHE_DIR="$SCRATCH_DIR/.uv-cache"
ENV="$SYNTH_ENV_DIR/qc"
STAMP="$ENV/.req-$(md5sum synth/envs/qc.txt | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
  rm -rf "$ENV"; uv venv --python 3.11 "$ENV"
  uv pip install --python "$ENV/bin/python" -r synth/envs/base.txt -r synth/envs/qc.txt
  touch "$STAMP"
fi
for d in "$@"; do "$ENV/bin/python" synth/qc.py "$d"; done
