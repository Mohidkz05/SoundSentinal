#!/bin/bash
# Download a LibriSpeech subset: test-clean by default (2,620 utterances, 40
# speakers, ~346 MB), or train-clean-100 (28,539 utterances, 251 speakers,
# ~6.3 GB) for synth/, which reads its transcripts and clones its speakers
# (RESULTS.md Finding 18).
#
#   sbatch hpc/get_librispeech.slurm                    # test-clean
#   sbatch hpc/get_librispeech.slurm train-clean-100    # synth/ prompts and text
#
# CALIBRATION data for Finding 15 (the clean-audio threshold), never training
# data — see the header of ai_model/librispeech.py. CC BY 4.0, from OpenSLR
# (resource 12). Checked against the MD5 OpenSLR publishes. After unpacking,
# the speaker list is compared with every LibriTTS speaker SpeechFake uses; the
# job fails if any appears, because then the set would not be held out.
#
#SBATCH --job-name=ls-download
#SBATCH --account=df37
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

ROOT="${LIBRISPEECH_ROOT:-$SCRATCH_DIR/librispeech}"
SUBSET="${1:-test-clean}"
TGZ="$ROOT/$SUBSET.tar.gz"
case "$SUBSET" in   # OpenSLR md5sum.txt, resource 12
  test-clean)      MD5="32fa31d27d2e1cad72775fee3f4849a9" ;;
  train-clean-100) MD5="2a93770f6d5c6c964bc36631d331a522" ;;
  *) echo "ERROR: unknown subset $SUBSET" >&2; exit 1 ;;
esac
mkdir -p "$ROOT"

if [ ! -f "$TGZ.ok" ]; then
  # Same resume loop as get_voxpopuli.slurm, and for the same reason no
  # curl --retry: it truncates a resumed file back to where it started.
  for attempt in $(seq 1 100); do
    curl -sSL -C - -Y 1000000 -y 60 --connect-timeout 60 -o "$TGZ" \
      "https://www.openslr.org/resources/12/$SUBSET.tar.gz" && break
    sleep 10
  done
  if [ "$(md5sum "$TGZ" | cut -d' ' -f1)" = "$MD5" ]; then
    touch "$TGZ.ok"; echo "== $SUBSET.tar.gz: $(du -h "$TGZ" | cut -f1), md5 ok =="
  else
    echo "ERROR: md5 mismatch; deleted, re-submit to refetch" >&2
    rm -f "$TGZ"; exit 1
  fi
fi

echo "== Extracting =="
tar -xzf "$TGZ" -C "$ROOT"
# Only test-clean must be speaker-disjoint from training; train-clean-100 is
# the same 251 speakers as SpeechFake's LibriTTS, on purpose.
if [ "$SUBSET" = test-clean ]; then
  (cd ai_model && "$VENV_DIR/bin/python" librispeech.py check "$ROOT")
fi
du -sh "$ROOT/LibriSpeech"
