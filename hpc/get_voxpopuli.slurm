#!/bin/bash
# Download VoxPopuli English (three shards, ~5 GB) and unpack it for calibration.
#
#   sbatch hpc/get_voxpopuli.slurm
#
# CALIBRATION data for the decision threshold, never training data — see the
# header of ai_model/voxpopuli.py and RESULTS.md Finding 9. CC0, ungated, from
# Hugging Face (facebook/voxpopuli). Fetched: en validation (1,753 clips),
# test (1,842) and the first of 30 train shards (~6,000), enough for the 10,000
# clips calibrate.py samples by default. Each file is checked against the
# SHA-256 Hugging Face publishes for it (the LFS oid).
#
#SBATCH --job-name=voxpopuli-download
#SBATCH --account=df37
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

BASE="https://huggingface.co/datasets/facebook/voxpopuli/resolve/main"
ROOT="${VOXPOPULI_ROOT:-$SCRATCH_DIR/voxpopuli}"
SHARDS="$ROOT/shards"
mkdir -p "$SHARDS"

# path sha256 — the LFS oids from the Hugging Face API, 27 September 2026.
PARTS="
en/validation-00000-of-00001.parquet bc8bbe5fe23aba60c3e02e6f6239943bc92abc0aa327c926bd3a2d61e44d646b
en/test-00000-of-00001.parquet 02cc7290425ddb95beeabda1d1e81e5e068cd7a25380bcf1fb69071b62610ffa
en/train-00000-of-00030.parquet 1b8c8c36caabc4796f42017ed10b78669369f4dce44a3df23f0bedd7b5c5051a
"

while read -r path sha; do
  [ -z "$path" ] && continue
  f="$SHARDS/$(basename "$path")"
  if [ -f "$f.ok" ]; then echo "== $path: verified earlier =="; continue; fi
  # Same resume loop as get_speechfake.slurm, and for the same reason no
  # curl --retry: it truncates a resumed file back to where it started.
  for attempt in $(seq 1 200); do
    curl -sSL -C - -Y 1000000 -y 60 --connect-timeout 60 -o "$f" "$BASE/$path" && break
    sleep 10
  done
  if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then
    touch "$f.ok"; echo "== $path: $(du -h "$f" | cut -f1), sha256 ok =="
  else
    echo "ERROR: $path sha256 mismatch; deleted, re-submit to refetch" >&2
    rm -f "$f"; exit 1
  fi
done <<< "$PARTS"

if [ ! -f "$ROOT/clips.csv" ]; then
  echo "== Extracting audio =="
  (cd ai_model && "$VENV_DIR/bin/python" voxpopuli.py extract "$SHARDS" "$ROOT")
fi
head -3 "$ROOT/clips.csv"
du -sh "$ROOT"
