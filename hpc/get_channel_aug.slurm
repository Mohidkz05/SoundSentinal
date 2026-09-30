#!/bin/bash
# Download the noise and room recordings behind --channel-aug (RESULTS.md
# Finding 16): MUSAN's noise and music (OpenSLR 17, CC BY 4.0) and the RIRS
# room impulse responses (OpenSLR 28, Apache 2.0). ~12.4 GB of archives.
#
#   sbatch hpc/get_channel_aug.slurm
#
# TRAINING-ONLY material: channel_aug.py mixes it into training clips, real
# and fake alike. No test or calibration set contains either corpus. MUSAN's
# speech subset is not extracted — mixing someone else's real voice into a
# fake clip would blur what the label means.
#
# OpenSLR publishes no MD5 for these two resources (checked 30 September
# 2026: md5sum.txt is a 404), so each archive is checked against the byte
# count its server reports, then tested whole before extraction.
#
#SBATCH --job-name=chaug-download
#SBATCH --account=df37
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

ROOT="${CHANNEL_AUG_ROOT:-$SCRATCH_DIR/channel_aug}"
mkdir -p "$ROOT"

fetch() {  # url file bytes
  local url=$1 out=$2 bytes=$3
  [ -f "$out.ok" ] && return 0
  # Same resume loop as get_librispeech.slurm, and for the same reason no
  # curl --retry: it truncates a resumed file back to where it started.
  for attempt in $(seq 1 100); do
    curl -sSL -C - -Y 1000000 -y 60 --connect-timeout 60 -o "$out" "$url" && break
    sleep 10
  done
  local got; got=$(stat -c %s "$out")
  if [ "$got" != "$bytes" ]; then
    echo "ERROR: $out is $got bytes, expected $bytes; deleted, re-submit" >&2
    rm -f "$out"; exit 1
  fi
  touch "$out.ok"; echo "== $(basename "$out"): $(du -h "$out" | cut -f1), size ok =="
}

fetch https://www.openslr.org/resources/17/musan.tar.gz "$ROOT/musan.tar.gz" 11086114085
fetch https://www.openslr.org/resources/28/rirs_noises.zip "$ROOT/rirs_noises.zip" 1311166223

echo "== Extracting MUSAN noise + music =="
tar -xzf "$ROOT/musan.tar.gz" -C "$ROOT" musan/noise musan/music
echo "== Extracting RIRS =="
unzip -tq "$ROOT/rirs_noises.zip"
unzip -oq "$ROOT/rirs_noises.zip" -d "$ROOT"

(cd ai_model && "$VENV_DIR/bin/python" channel_aug.py check "$ROOT")
du -sh "$ROOT"/musan "$ROOT"/RIRS_NOISES
