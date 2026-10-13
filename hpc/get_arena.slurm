#!/bin/bash
# Fetch one Speech Anti-Spoofing Arena test set and unpack it for
# ai_model/arena.py. EVALUATION ONLY — see that file's header.
#
#   sbatch hpc/get_arena.slurm ASVspoof2021_LA                    # 7.6 GB
#   sbatch --time=06:00:00 hpc/get_arena.slurm ASVspoof2021_DF    # 34 GB
#
# Ungated, from Hugging Face (SpeechAntiSpoofingBenchmarks/<set>), at the
# revision the Arena's manifest pins (arena.SETS), so the trials are the
# leaderboard's. Only sets listed there can be fetched: the others are
# non-commercial. Each shard is checked against the SHA-256 Hugging Face
# publishes for it (the LFS oid) and deleted once its audio is extracted.
#
#SBATCH --job-name=arena-download
#SBATCH --account=df37
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

NAME="${1:?usage: sbatch hpc/get_arena.slurm <set name, see ai_model/arena.py>}"
PY="$VENV_DIR/bin/python"
REV="$(cd ai_model && "$PY" -c "import arena, sys; print(arena.SETS[sys.argv[1]][0])" "$NAME")"
REPO="SpeechAntiSpoofingBenchmarks/$NAME"
ROOT="$ARENA_ROOT/$NAME"
SHARDS="$ROOT/shards"
mkdir -p "$SHARDS"
echo "== $REPO @ $REV -> $ROOT =="

if [ -f "$ROOT/trials.csv" ]; then
  echo "== already extracted: $(($(wc -l < "$ROOT/trials.csv") - 1)) trials =="; exit 0
fi

# path sha256, from the Hugging Face API at the pinned revision.
PARTS="$(curl -sSL --retry 5 "https://huggingface.co/api/datasets/$REPO/tree/$REV/data" | "$PY" -c "
import json, sys
for f in json.load(sys.stdin):
    if f['path'].rsplit('/', 1)[-1].startswith('test-') and f['path'].endswith('.parquet'):
        print(f['path'], f['lfs']['oid'])")"
echo "== $(echo "$PARTS" | wc -l) shards =="

while read -r path sha; do
  [ -z "$path" ] && continue
  f="$SHARDS/$(basename "$path")"
  if [ -f "$f.ok" ]; then echo "== $path: verified earlier =="; continue; fi
  # Same resume loop as get_peoples_speech.slurm, and for the same reason no
  # curl --retry: it truncates a resumed file back to where it started.
  for attempt in $(seq 1 200); do
    curl -sSL -C - -Y 1000000 -y 60 --connect-timeout 60 -o "$f" \
      "https://huggingface.co/datasets/$REPO/resolve/$REV/$path" && break
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
(cd ai_model && "$PY" arena.py extract "$NAME" "$SHARDS" "$ROOT")
head -3 "$ROOT/trials.csv"
# The shards are a second copy of every clip; scratch is not for keeping both.
rm -rf "$SHARDS"
du -sh "$ROOT"
