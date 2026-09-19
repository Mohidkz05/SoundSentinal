#!/bin/bash
# Train on one GPU. Arguments are forwarded to train_dp_avspoof.py:
#
#   sbatch hpc/train.slurm --no-dp        # the non-private baseline — do this first
#   sbatch hpc/train.slurm                # the DP arm
#   sbatch hpc/train.slurm --no-dp --no-class-weights
#
# Per APPROACH.md the baseline comes first: it is the accuracy ceiling, and the
# gap between it and the DP run is the number the project exists to report.
#
#SBATCH --job-name=soundsentinal
#SBATCH --account=df37
#SBATCH --partition=m3g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# --partition=m3g is M3's GPU partition, but confirm it and the GPU types
# available to df37 with `show_cluster` before the first submit; some GPUs want
# an explicit type, e.g. --gres=gpu:V100:1. A wrong partition name fails at
# submit, which is the harmless way to find out.
#
# --cpus-per-task=8 is not incidental: train_dp_avspoof.py defaults its
# DataLoader workers to $SLURM_CPUS_PER_TASK, and at 267k parameters the
# dataloader is the bottleneck rather than the GPU. Raising this is the single
# most effective change to wall-clock time. Raise --mem with it; workers fork.
#
# --time=02:00:00 is generous for the current CNN (5 epochs measured at ~10 min
# on CPU alone). Keep it honest — a shorter request schedules sooner.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "   submitted from : $PWD"
echo "   commit         : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "   cpus-per-task  : ${SLURM_CPUS_PER_TASK:-unset}"
echo "   ASVSPOOF_ROOT  : $ASVSPOOF_ROOT"
echo "   CKPT_ROOT      : $CKPT_ROOT"
echo "   args           : ${*:-<none, DP run>}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "   WARNING: no nvidia-smi — this job has no GPU and will train on CPU"
echo

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "ERROR: no venv at $VENV_DIR. Run: bash hpc/bootstrap.sh" >&2
  exit 1
fi
if [ ! -d "$ASVSPOOF_ROOT/LA/ASVspoof2019_LA_train/flac" ]; then
  echo "ERROR: no corpus at $ASVSPOOF_ROOT/LA. Run: sbatch hpc/get_la.slurm" >&2
  exit 1
fi

cd ai_model

# The train/serve parity check, before spending GPU hours. It asserts the tensor
# app.py builds is byte-identical to the one the training dataset builds — the
# exact failure that once made this repo's predictions quietly wrong for months.
echo "== verify_setup.py =="
"$VENV_DIR/bin/python" verify_setup.py
echo

echo "== train_dp_avspoof.py =="
"$VENV_DIR/bin/python" train_dp_avspoof.py --corpus LA "$@"

echo
echo "== Checkpoints =="
ls -lh "$CKPT_ROOT" "$CKPT_ROOT/nodp" 2>/dev/null || true
cat <<MSG

Copy the winning weights back to the machine that serves, from your laptop:
  scp mkha0155@m3-dtn.massive.org.au:$CKPT_ROOT/nodp/best.pth ai_model/checkpoints/
MSG
