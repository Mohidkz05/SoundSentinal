#!/bin/bash
# Score every per-epoch checkpoint of a run on the eval partition.
#
#   sbatch hpc/sweep_epochs.slurm                    # the non-private log-Mel run
#   sbatch hpc/sweep_epochs.slurm checkpoints/lfcc/nodp
#
# WHY: train_dp_avspoof.py selects best.pth by *dev* EER, and dev reuses the six
# attacks seen in training. The baseline scored 0.24% on dev and 10.15% on eval,
# a factor of 42 — so the selection criterion may be picking the wrong epoch
# entirely, and every future comparison would inherit that. This answers whether
# dev EER tracks eval EER at all. It needs no retraining: the trainer already
# writes a timestamped checkpoint every epoch.
#
#SBATCH --job-name=ss-sweep
#SBATCH --account=df37
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

RUN_DIR="${1:-$CKPT_ROOT/nodp}"
[ -d "$RUN_DIR" ] || RUN_DIR="$CKPT_ROOT/$1"

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "   run dir : $RUN_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader || echo "   WARNING: no GPU"
echo

cd ai_model
shopt -s nullglob
CKPTS=("$RUN_DIR"/deepfake_*.pth)
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "ERROR: no per-epoch checkpoints (deepfake_*.pth) in $RUN_DIR" >&2
  exit 1
fi
echo "Found ${#CKPTS[@]} per-epoch checkpoints."
echo

for ck in "${CKPTS[@]}"; do
  # Name the output by epoch rather than by wall-clock, so re-running overwrites
  # the same file instead of accumulating near-duplicates.
  ep=$("$VENV_DIR/bin/python" -c "
import torch,sys
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch','x'))" "$ck")
  echo "=== epoch $ep : $(basename "$ck") ==="
  "$VENV_DIR/bin/python" evaluate.py --corpus LA --ckpt "$ck" \
      --out "$RUN_DIR/eval_epoch$(printf '%02d' "$ep").json" \
    | grep -E "min t-DCF|^EER|front-end|dev EER" || true
  echo
done

echo "== Summary =="
"$VENV_DIR/bin/python" summarise_results.py "$RUN_DIR" --sort epoch
