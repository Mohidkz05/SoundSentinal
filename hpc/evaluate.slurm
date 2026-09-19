#!/bin/bash
# Score a trained checkpoint on the ASVspoof2019 eval partition.
#
#   sbatch hpc/evaluate.slurm              # the --no-dp baseline
#   sbatch hpc/evaluate.slurm --dp         # the DP run
#   sbatch hpc/evaluate.slurm --partition dev   # reproduce a training run's own number
#
# Arguments are forwarded to evaluate.py. This is the number that goes in a
# writeup: eval holds attacks A07–A19, none of them seen in training, and every
# row of the comparison table in APPROACH.md is an eval number.
#
#SBATCH --job-name=ss-eval
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# --time=00:30:00: eval is 71,933 clips against dev's 24,986, roughly three
# times the work of one training epoch's evaluation, and there is no backward
# pass. Training all five epochs took 2m49s, so half an hour is generous.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "   commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "   args   : ${*:-<none, scoring the non-private baseline on eval>}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || \
  echo "   WARNING: no nvidia-smi — this job has no GPU"
echo

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "ERROR: no venv at $VENV_DIR. Run: bash hpc/bootstrap.sh" >&2
  exit 1
fi
if [ ! -d "$ASVSPOOF_ROOT/LA/ASVspoof2019_LA_eval/flac" ]; then
  echo "ERROR: no eval partition at $ASVSPOOF_ROOT/LA. Run: sbatch hpc/get_la.slurm" >&2
  exit 1
fi

cd ai_model
"$VENV_DIR/bin/python" evaluate.py --corpus LA "$@"

echo
echo "== Result files =="
ls -lt "$CKPT_ROOT"/nodp/eval_*.json "$CKPT_ROOT"/eval_*.json 2>/dev/null | head -5 || true
