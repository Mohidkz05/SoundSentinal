#!/bin/bash
# Download VoxPopuli English (eight shards, ~24 GB) and unpack it.
#
#   sbatch hpc/get_voxpopuli.slurm
#
# CC0, ungated, from Hugging Face (facebook/voxpopuli). Two uses, see the
# header of ai_model/voxpopuli.py:
#   - calibration (Findings 9, 10): en validation (1,753 clips), test (1,842)
#     and train shard 00000 (~6,000) — the first three fetched;
#   - training (Finding 11, --extra-bonafide commonvoice+voxpopuli): train
#     shards 00000-00005, ~36k clips, added 28 September 2026.
# Each file is checked against the SHA-256 Hugging Face publishes for it (the
# LFS oid). Re-running skips verified shards and audio already extracted, and
# rewrites clips.csv to cover every shard present.
#
#SBATCH --job-name=voxpopuli-download
#SBATCH --account=df37
#SBATCH --time=06:00:00
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
en/train-00001-of-00030.parquet 7948745b67871656b5a06564a663b69c8e7ea342062517f95968cb47171df4dd
en/train-00002-of-00030.parquet 5f82edd14a55c9e89d1d9bc78255594cbb277e8a25f9156ab53633f5898654b6
en/train-00003-of-00030.parquet 530c9b1aa724fe204a5cc03c468d003c7c961d2c48f3dc5eef72d648718b3b5d
en/train-00004-of-00030.parquet 80f1fef7d241f087d7919d1bd0fd8bb0e654b1f82bb21a1941124e8953d6d88e
en/train-00005-of-00030.parquet e577ea70be34a778ddab1f5c4a1932816ffbdbb86c1b804f43993f4e722f5fce
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

echo "== Extracting audio =="
(cd ai_model && "$VENV_DIR/bin/python" voxpopuli.py extract "$SHARDS" "$ROOT")
head -3 "$ROOT/clips.csv"
du -sh "$ROOT"
