#!/bin/bash
# Build the training environment on M3. Safe to re-run; it is idempotent.
#
#   cd ~/SoundSentinal && bash hpc/bootstrap.sh
#
# Runs on a login node — it is I/O and network bound, not CPU bound, so it does
# not need a job. Takes a few minutes, most of it downloading CUDA wheels.
#
# Deliberately does not touch `module load`. uv fetches its own CPython, and the
# torch wheels bundle the CUDA runtime, so there is no module to guess at and
# nothing to re-load inside every job script. The only thing that must come from
# the system is the NVIDIA driver, which is on the GPU nodes already.
set -euo pipefail

cd "$(dirname "$0")/.."
source hpc/env.sh

echo "== Target layout =="
printf '  %-14s %s\n' repo "$PWD" venv "$VENV_DIR" data "$ASVSPOOF_ROOT" \
                      checkpoints "$CKPT_ROOT" logs "$LOG_DIR"

# --- uv ---------------------------------------------------------------------
# Chosen over python3 -m venv because it brings its own interpreter. M3's system
# python is whatever the login node ships, and the ensurepip/sudo dead end
# documented in CLAUDE.md is exactly the kind of thing there is no fixing
# without root on a shared machine.
if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv into $UV_BIN_DIR =="
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
  PATH="$UV_BIN_DIR:$PATH"
fi
echo "uv: $(uv --version)"

# --- venv -------------------------------------------------------------------
# 3.12 matches the local venv, so a checkpoint or a pickle moves between the
# two without a version question.
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "== Creating venv at $VENV_DIR =="
  uv venv --python 3.12 "$VENV_DIR"
fi

echo "== Installing dependencies (CUDA build) =="
VIRTUAL_ENV="$VENV_DIR" uv pip install -r hpc/requirements-cuda.txt

# --- verify -----------------------------------------------------------------
# On a login node there is no GPU, so `cuda.is_available()` is expected to be
# False here. What this checks is that a CUDA-built torch imported at all; the
# device check that matters happens in the job, where train_dp_avspoof.py
# prints the device it got.
echo "== Verifying =="
"$VENV_DIR/bin/python" - <<'PY'
import torch, torchaudio, soundfile, opacus, numpy, pandas
print(f"torch       {torch.__version__}")
print(f"torchaudio  {torchaudio.__version__}")
print(f"opacus      {opacus.__version__}")
print(f"soundfile   {soundfile.__version__}")
built = torch.version.cuda
print(f"CUDA build  {built or 'NONE — this is a CPU wheel, the index URL did not take'}")
print(f"GPU visible {torch.cuda.is_available()}  (False is correct on a login node)")
assert built, "installed a CPU-only torch; check hpc/requirements-cuda.txt"
PY

cat <<MSG

== Done ==
Next:
  sbatch hpc/get_la.slurm      # download + extract ASVspoof2019 LA (~7.6GB)
  sbatch hpc/train.slurm       # once the data is there

Confirm the GPU partition and its name before the first train submit:
  show_cluster
  sinfo -s
  user_info                    # quota on the three project directories
MSG
