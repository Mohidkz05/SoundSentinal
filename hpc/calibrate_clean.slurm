#!/bin/bash
# Add a clean-audio threshold route to a calibrated checkpoint (Finding 15).
#
#   sbatch --gres=gpu:L40S:1 \
#          --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
#          hpc/calibrate_clean.slurm --ckpt <People's Speech-calibrated>.pth \
#          [--target-frr 0.01]
#
# Arguments are forwarded to ai_model/calibrate_clean.py, which explains why
# (a third of clean fakes pass the served threshold, RESULTS.md Finding 13).
# Needs People's Speech (hpc/get_peoples_speech.slurm) and LibriSpeech
# test-clean (hpc/get_librispeech.slurm). Writes <stem>_routed.pth beside the
# input; score it with evaluate.slurm --ckpt <that file>.
#
#SBATCH --job-name=ss-calclean
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "   commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "   args   : $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || \
  echo "   WARNING: no nvidia-smi — this job has no GPU"
echo

cd ai_model
"$VENV_DIR/bin/python" calibrate_clean.py "$@"
