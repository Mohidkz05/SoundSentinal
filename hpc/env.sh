# Shared paths for the M3 (MASSIVE) account. Sourced by every other script in
# this directory and by any interactive session — `source hpc/env.sh`.
#
# Nothing here is specific to a login node vs a compute node, so the same file
# is correct in a batch job, an smux session and at an ssh prompt.

PROJECT_ID="df37"

# The three directories M3 links into $HOME. Their roles are not
# interchangeable: primary is backed up and small, scratch is large and holds
# anything that can be regenerated from a URL or a script.
PROJECT_DIR="$HOME/${PROJECT_ID}"
SCRATCH_DIR="$HOME/${PROJECT_ID}_scratch"

# The clone itself. $HOME has the tightest quota on M3, so the repo (a few MB)
# is the only thing that belongs there.
REPO_DIR="${REPO_DIR:-$HOME/SoundSentinal}"

# The venv is ~5GB with CUDA torch, which is why it is NOT in $HOME.
VENV_DIR="$PROJECT_DIR/venv"
UV_BIN_DIR="$PROJECT_DIR/bin"
export UV_CACHE_DIR="$PROJECT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.uv-python"

# The corpus: 7.6GB compressed, ~11GB extracted, and re-downloadable from
# Edinburgh DataShare — textbook scratch data.
export ASVSPOOF_ROOT="$SCRATCH_DIR/data"

# Checkpoints are small and are the actual output of a run, so they go to
# primary storage. train_dp_avspoof.py and app.py both read this variable.
export CKPT_ROOT="$PROJECT_DIR/checkpoints"

# Slurm logs. Created here so a job never fails because its --output path
# does not exist yet, which Slurm reports as a bare "Batch job submit failed".
export LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$ASVSPOOF_ROOT"

# Put uv and the venv on PATH if they exist yet (they will not on first run).
[ -d "$UV_BIN_DIR" ] && PATH="$UV_BIN_DIR:$PATH"
[ -d "$VENV_DIR/bin" ] && PATH="$VENV_DIR/bin:$PATH"
export PATH
