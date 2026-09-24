#!/bin/bash
# Train on one GPU. Arguments are forwarded to train_dp_avspoof.py:
#
#   sbatch hpc/train.slurm --no-dp        # the CNN baseline — do this first
#   sbatch hpc/train.slurm                # the CNN DP arm
#   sbatch hpc/train.slurm --no-dp --no-class-weights
#
# Per APPROACH.md the baseline comes first: it is the accuracy ceiling, and the
# gap between it and the DP run is the number the project exists to report.
#
# --- AASIST needs a longer wall clock than the default below -----------------
#
#   sbatch --time=12:00:00 hpc/train.slurm --arch aasist --no-dp
#   sbatch --time=12:00:00 hpc/train.slurm --arch aasist            # DP arm
#   sbatch --time=12:00:00 hpc/train.slurm --arch aasist-l --no-dp  # if memory is tight
#   sbatch --time=14:00:00 hpc/train.slurm --arch aasist --no-dp --rawboost 5  # augmented
#   sbatch --time=16:00:00 hpc/train.slurm --arch aasist --no-dp \
#          --extra-train asvspoof5 --epochs 12       # + ASVspoof 5 train (~8x the data)
#   sbatch --partition=m3h --qos=m3h --gres=gpu:H100:1 --time=24:00:00 \
#          hpc/train.slurm --arch ssl-aasist --no-dp  # XLS-R + AASIST, 316M params
#     (needs a big GPU: fine-tuning XLS-R at batch 14 will not fit a 16GB T4.
#      Constant lr, so a run cut off at its limit resumes with the same command.)
#
# A command-line --time overrides the #SBATCH directive, so the default stays
# honest for the CNN rather than every small job asking for half a day.
#
# Why AASIST is a different size of job: 100 epochs at batch 24 (its published
# recipe — see TRAIN_DEFAULTS in train_dp_avspoof.py for why neither number is
# a free choice) against the CNN's 5 epochs at batch 64. Twelve hours is a
# guess, not a measurement — nothing has timed this yet. It does not need to be
# right: train_dp_avspoof.py auto-resumes from last.pth, and the cosine
# schedule is checkpointed with it, so a job killed at its limit is continued
# by resubmitting exactly the same command. Read the epoch number in the log
# and extrapolate before asking for more.
#
# Checkpoints go to $CKPT_ROOT/<arch>/[<frontend>/][nodp/], so runs never
# collide and nothing auto-resumes from the wrong weights.
#
#SBATCH --job-name=soundsentinal
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# --partition=gpu, confirmed against this account on 19 September 2026:
# AllowAccounts=ALL, AllowQos=ALL, MaxTime 7 days, 40 nodes / 132 GPUs holding
# A100-80G, L40S, A40 and T4. There is NO m3g partition — `m3g[100-119]` are the
# L40S *node names* inside `gpu`, which is an easy and confusing thing to
# mistake for a partition.
#
# Plain --gres=gpu:1 takes whichever type frees up first, which is right for a
# 267k-parameter CNN that fits on any of them. Name a type only when a model
# actually needs it, e.g. --gres=gpu:A100:1 for 80GB.
#
# For the wav2vec2 / WavLM ceiling row in APPROACH.md there is something better:
# df37 holds the `m3h` QOS, which unlocks the H100 partition.
#   #SBATCH --partition=m3h
#   #SBATCH --qos=m3h
#   #SBATCH --gres=gpu:H100:1
# Do not use it for the small CNN — it is a scarce resource (4 nodes) and this
# model cannot use it.
#
# --cpus-per-task=8 is not incidental: train_dp_avspoof.py defaults its
# DataLoader workers to $SLURM_CPUS_PER_TASK, and for the CNN the dataloader is
# the bottleneck rather than the GPU. Raising this is the single most effective
# change to wall-clock time. Raise --mem with it; workers fork.
#
# For AASIST that is no longer true — it does far more arithmetic per clip, so
# the GPU is the bottleneck and more workers buy little. What AASIST does want
# is memory: its first residual block holds a (batch, 32, 24, 21290) activation,
# which is why its batch size is 24 and not 64.
#
# --time=02:00:00 is generous for the current CNN (5 epochs measured at ~10 min
# on CPU alone). Keep it honest — a shorter request schedules sooner, and the
# partition would otherwise default to a full day.

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
find "$CKPT_ROOT" -name "best.pth" -o -name "last.pth" | sort || true
cat <<MSG

Copy the winning weights back to the machine that serves, from your laptop.
The CNN baseline:
  scp mkha0155@m3-dtn.massive.org.au:$CKPT_ROOT/nodp/best.pth ai_model/checkpoints/
AASIST:
  scp mkha0155@m3-dtn.massive.org.au:$CKPT_ROOT/aasist/nodp/best.pth ai_model/checkpoints/
app.py reads checkpoints/best.pth whatever produced it — the architecture and
the front-end travel inside the file.
MSG
