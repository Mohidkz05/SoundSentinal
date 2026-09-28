#!/bin/bash
# Download People's Speech `clean` test (nine shards, ~4 GB) and unpack it.
#
#   sbatch hpc/get_peoples_speech.slurm
#
# CALIBRATION data for Finding 11, never training data — see the header of
# ai_model/peoples_speech.py. Ungated, from Hugging Face
# (MLCommons/peoples_speech). Only `clean` is fetched: it is CC-BY, which
# allows commercial use; the `_sa` configurations are CC-BY-SA and are not.
# 34,898 clips. Each file is checked against the SHA-256 Hugging Face
# publishes for it (the LFS oid). The extract step prints every recording
# whose name matches an In-the-Wild speaker; those are excluded from
# calibration.
#
#SBATCH --job-name=ps-download
#SBATCH --account=df37
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

BASE="https://huggingface.co/datasets/MLCommons/peoples_speech/resolve/main"
ROOT="${PEOPLES_SPEECH_ROOT:-$SCRATCH_DIR/peoples_speech}"
SHARDS="$ROOT/shards"
mkdir -p "$SHARDS"

# path sha256 — the LFS oids from the Hugging Face API, 28 September 2026.
PARTS="
clean/test-00000-of-00009.parquet 6cae812dd9f6a1e027ae7f672d7cb7c756253078d25b5d9eb4ae6171eb68fdf0
clean/test-00001-of-00009.parquet 13eaf1b41f0ff5f07a45ef006f801998bbc3c5b66d88c38086fd81a498166dea
clean/test-00002-of-00009.parquet 70f417389c5274cbd77ff6da2af141639bbcc07e3e6e56a9eba9b73daf3fcc22
clean/test-00003-of-00009.parquet 9e2c7970f80726ace98c5956e2b40a9d34cf9f08a47a47fba9c271830bb4f430
clean/test-00004-of-00009.parquet 3fd7179276e902783cf7fbb453f9df74f43148b567f52b2c9b5210e1ff2d2851
clean/test-00005-of-00009.parquet 5c26e4c18fa19a02986d9b9cac6f847ca6d8237e28059380ee3e48ff5ccfa3f8
clean/test-00006-of-00009.parquet 5665360504eade7cd6cb4886cc912348e229e70bd0de28fe3be78a3c734359fe
clean/test-00007-of-00009.parquet 481a79a5b83238a5dad20f5f7b3f980a98af4fe5fdd0cb6f1c2a30f68c792d13
clean/test-00008-of-00009.parquet 6e9fcbe2a6d5c321621c5b49cac5da569d0540585fcd52415bff66f1abf2995a
"

while read -r path sha; do
  [ -z "$path" ] && continue
  f="$SHARDS/$(basename "$path")"
  if [ -f "$f.ok" ]; then echo "== $path: verified earlier =="; continue; fi
  # Same resume loop as get_voxpopuli.slurm, and for the same reason no
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
(cd ai_model && "$VENV_DIR/bin/python" peoples_speech.py extract "$SHARDS" "$ROOT")
head -3 "$ROOT/clips.csv"
du -sh "$ROOT"
