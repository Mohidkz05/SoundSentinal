#!/bin/bash
# Score a trained checkpoint on the ASVspoof2019 eval partition.
#
#   sbatch hpc/evaluate.slurm                        # the CNN --no-dp baseline
#   sbatch hpc/evaluate.slurm --dp                   # the CNN DP run
#   sbatch hpc/evaluate.slurm --arch aasist          # AASIST, non-private
#   sbatch hpc/evaluate.slurm --arch aasist --dp     # AASIST, DP
#   sbatch hpc/evaluate.slurm --frontend lfcc        # the CNN LFCC ablation
#   sbatch hpc/evaluate.slurm --partition dev        # reproduce a run's own number
#   sbatch hpc/evaluate.slurm --dataset itw --arch aasist   # In-the-Wild
#
# --dataset itw scores In-the-Wild instead of ASVspoof: 31,779 real-world clips
# from 58 public figures, deepfakes people actually posted rather than lab
# attacks generated to a protocol. Needs hpc/get_in_the_wild.slurm first. EER
# only — min t-DCF needs the organisers' ASV scores, which ship only with
# ASVspoof. Expect a far worse number than LA eval; that gap is the result.
#
# --arch and --frontend pick the checkpoint directory, so the path never has to
# be typed; --ckpt still overrides both for anything else.
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
# pass. Training all five CNN epochs took 2m49s, so half an hour is generous
# there. AASIST is a much heavier forward pass and has not been timed — if this
# job hits the limit, resubmit with --time=02:00:00 rather than assuming a hang.
#
# Plain --gres=gpu:1 can land on a 16GB T4, where AASIST at batch 128 runs out
# of memory (job 60384454, 24 September 2026). For AASIST pass
# --gres=gpu:L40S:1 (or A40/A100) on the sbatch line.

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
# Which corpus this run needs depends on --dataset, and In-the-Wild needs no
# ASVspoof at all — checking for the LA eval partition regardless would block a
# perfectly valid job on a machine that only has In-the-Wild.
case " $* " in
  *" --dataset itw "*)
    if [ ! -f "$ITW_ROOT/release_in_the_wild/meta.csv" ]; then
      echo "ERROR: no In-the-Wild at $ITW_ROOT. Run: sbatch hpc/get_in_the_wild.slurm" >&2
      exit 1
    fi
    ;;
  *)
    if [ ! -d "$ASVSPOOF_ROOT/LA/ASVspoof2019_LA_eval/flac" ]; then
      echo "ERROR: no eval partition at $ASVSPOOF_ROOT/LA. Run: sbatch hpc/get_la.slurm" >&2
      exit 1
    fi
    ;;
esac

cd ai_model
"$VENV_DIR/bin/python" evaluate.py --corpus LA "$@"

echo
echo "== Result files =="
find "$CKPT_ROOT" -name "eval_*.json" -newermt "-1 hour" 2>/dev/null | sort || true
