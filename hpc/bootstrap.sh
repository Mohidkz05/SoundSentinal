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
  # INSTALLER_NO_MODIFY_PATH=1 is not optional here. Without it the installer
  # appends `. "$UV_BIN_DIR/env"` to .bashrc, .bash_profile AND .profile. That
  # is rude on a shared account, and it breaks loudly the moment the directory
  # moves: every login and every non-interactive ssh command then prints
  # "No such file or directory" before doing anything else, which is enough to
  # corrupt the output of scripts that parse what they get back over ssh.
  # env.sh already puts uv on PATH, so the shell config never needed touching.
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_BIN_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
  PATH="$UV_BIN_DIR:$PATH"
fi
echo "uv: $(uv --version)"

# --- venv -------------------------------------------------------------------
# 3.12 matches the local venv, so a checkpoint or a pickle moves between the
# two without a version question.
# -x follows the symlink, so this is false for a venv whose interpreter link
# dangles — which is what a moved venv looks like. Rebuilding is the fix; a venv
# records absolute paths and does not survive being relocated.
if [ ! -x "$VENV_DIR/bin/python" ]; then
  [ -d "$VENV_DIR" ] && { echo "== Removing stale venv at $VENV_DIR =="; rm -rf "$VENV_DIR"; }
  echo "== Creating venv at $VENV_DIR =="
  uv venv --python 3.12 "$VENV_DIR"
fi

# Two phases, one index each. See the header of requirements-torch.txt — a
# single file naming both indexes fails whichever order they are given in.
echo "== Installing torch/torchaudio (CUDA build) =="
VIRTUAL_ENV="$VENV_DIR" uv pip install -r hpc/requirements-torch.txt

echo "== Installing everything else (PyPI) =="
VIRTUAL_ENV="$VENV_DIR" uv pip install -r hpc/requirements-rest.txt

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
assert built, "installed a CPU-only torch; check hpc/requirements-torch.txt"
PY

cat <<MSG

== Done ==
Next:
  sbatch hpc/get_la.slurm      # download + extract ASVspoof2019 LA (~7.6GB)
  sbatch hpc/train.slurm       # once the data is there

The partition and GPU names are already settled — see "What this account
actually has" in hpc/README.md. Handy anyway:
  user_info                    # quota on the three project directories
  squeue -u $USER              # what you have queued
MSG
