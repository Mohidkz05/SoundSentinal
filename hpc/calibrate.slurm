#!/bin/bash
# Re-derive a checkpoint's decision threshold from Common Voice bona fide speech.
#
#   sbatch --gres=gpu:L40S:1 \
#          --export=ALL,CKPT_ROOT=$HOME/df37_scratch/$USER/checkpoints \
#          hpc/calibrate.slurm --ckpt <path>/best.pth [--target-frr 0.05]
#
# Arguments are forwarded to ai_model/calibrate.py, which explains why (the
# dev-EER threshold flags 46% of genuine In-the-Wild clips, RESULTS.md
# Finding 9). Needs Common Voice: sbatch --export=ALL,WITH_COMMONVOICE=1
# hpc/get_speechfake.slurm. Writes <stem>_calibrated.pth beside the input;
# score In-the-Wild once with evaluate.slurm --ckpt <that file>.
#
#SBATCH --job-name=ss-calibrate
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
"$VENV_DIR/bin/python" calibrate.py "$@"
