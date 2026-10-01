#!/bin/bash
# Generate one model's fakes for one split (RESULTS.md Finding 18).
#
#   sbatch hpc/synth.slurm <family> <model> <split> <n>
#   sbatch hpc/synth.slurm kokoro kokoro-82m train 6000
#   sbatch --time=00:30:00 hpc/synth.slurm kokoro kokoro-82m train 3   # smoke test
#
# Each family gets its own uv environment on scratch, built on first use from
# synth/envs/<family>.txt, because the generators pin incompatible torch and
# transformers versions. Model weights download to $SYNTH_HF_HOME, also on
# scratch; every repo in ai_model/synth.py is ungated and commercially licensed.
# Resumable: resubmit the same line after a time-out.
#
#SBATCH --job-name=synth
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
FAMILY=$1 MODEL=$2 SPLIT=$3 N=$4

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh
export HF_HOME="$SYNTH_HF_HOME"
mkdir -p "$SYNTH_ROOT" "$SYNTH_ENV_DIR" "$HF_HOME"

echo "== Job ${SLURM_JOB_ID:-local} on $(hostname): $FAMILY / $MODEL / $SPLIT / $N =="
echo "   commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ENV="$SYNTH_ENV_DIR/$FAMILY"
REQ="synth/envs/$FAMILY.txt"
STAMP="$ENV/.req-$(md5sum "$REQ" | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
  echo "== Building $ENV from $REQ =="
  rm -rf "$ENV"
  PYVER=$(sed -n 's/^# python: *//p' "$REQ"); PYVER=${PYVER:-3.11}
  uv venv --python "$PYVER" "$ENV"
  export UV_CACHE_DIR="$SCRATCH_DIR/.uv-cache"
  # Base first, so packages that import numpy in setup.py (pkuseg, for
  # Chatterbox) can be built against it: list them on a "# no-build-isolation:"
  # line in the family's requirements file.
  uv pip install --python "$ENV/bin/python" -r synth/envs/base.txt setuptools wheel cython
  NBI=$(sed -n 's/^# no-build-isolation: *//p' "$REQ")
  uv pip install --python "$ENV/bin/python" -r synth/envs/base.txt -r "$REQ" \
    $(for p in $NBI; do echo --no-build-isolation-package "$p"; done)
  touch "$STAMP"
fi

cd synth
"$ENV/bin/python" generate.py --family "$FAMILY" --model "$MODEL" --split "$SPLIT" --n "$N"
