#!/bin/bash
# Download and extract ASVspoof2019 LA into scratch.
#
#   sbatch hpc/get_la.slurm
#
# Submitted as a job rather than run at the login prompt because it is the
# slowest step in the whole project (7.6GB in, 121,461 FLACs out) and login
# nodes are for "light-weight account housekeeping" per the M3 welcome email.
# No GPU, one core — this is entirely network and filesystem bound.
#
# Re-running is safe and is the intended way to resume: curl -C - continues a
# partial file, and both stages are skipped if their output is already complete.
#
#SBATCH --job-name=la-download
#SBATCH --account=df37
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
#
# No --partition: the cluster default is fine for a CPU job. If the default
# partition's time limit is under 8h the submit is rejected with a message
# saying so — check `sinfo -s` for the limits and name a longer partition.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

# The bitstream URL, not the /handle/ or /bitstream/handle/ path. DataShare
# moved to DSpace 7 and the old URLs that circulate in papers and READMEs now
# serve a 4KB HTML redirect page with Content-Type text/html — a "download"
# that succeeds, produces a file, and fails at unzip. Resolved from the DSpace
# REST API and verified to return `Content-Type: application/zip` with
# `Accept-Ranges: bytes`, which is what makes the resume below work.
LA_URL="https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
LA_BYTES=7640952520

ZIP="$SCRATCH_DIR/LA.zip"
mkdir -p "$SCRATCH_DIR" "$ASVSPOOF_ROOT"

echo "== Disk before =="
df -h "$SCRATCH_DIR" | tail -1
command -v user_info >/dev/null 2>&1 && user_info || true

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
have=0
[ -f "$ZIP" ] && have=$(stat -c %s "$ZIP")

if [ "$have" -eq "$LA_BYTES" ]; then
  echo "== LA.zip already complete ($LA_BYTES bytes), skipping download =="
else
  # Fail early and legibly if compute nodes have no route out. Some clusters
  # only allow outbound HTTP from the login and data-transfer nodes, in which
  # case this job cannot do its job and the fallback is the DTN (see README).
  if ! curl -sS -I --max-time 30 "$LA_URL" >/dev/null 2>&1; then
    echo "ERROR: cannot reach datashare.ed.ac.uk from this node." >&2
    echo "       Compute nodes may have no outbound route. Run the download on" >&2
    echo "       the data transfer node instead — see 'No internet' in hpc/README.md." >&2
    exit 1
  fi
  echo "== Downloading LA.zip ($(numfmt --to=iec $LA_BYTES)) =="
  [ "$have" -gt 0 ] && echo "   resuming from $(numfmt --to=iec "$have")"
  echo "   (no progress meter by design — watch the file grow instead:"
  echo "    ssh m3 'ls -lh $ZIP')"
  # --no-progress-meter, for the same reason tqdm gets disable=None in the
  # trainer: with no terminal attached curl writes its meter as one line per
  # update, and 25 minutes of that buries every other line in the log. -sS
  # would also hide the meter but silences errors with it; this keeps them.
  curl -L -C - --no-progress-meter --retry 10 --retry-delay 15 --retry-connrefused \
       -o "$ZIP" "$LA_URL"

  got=$(stat -c %s "$ZIP")
  if [ "$got" -ne "$LA_BYTES" ]; then
    echo "ERROR: LA.zip is $got bytes, expected $LA_BYTES." >&2
    echo "       Re-submit to resume; delete the file first if it is HTML." >&2
    exit 1
  fi
  echo "   size verified: $got bytes"
fi

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
# The archive's own top level is LA/, confirmed from its first local file
# header, so extracting into $ASVSPOOF_ROOT lands the tree exactly where
# get_corpus_paths() in train_dp_avspoof.py looks:
#   $ASVSPOOF_ROOT/LA/ASVspoof2019_LA_train/flac
#   $ASVSPOOF_ROOT/LA/ASVspoof2019_LA_cm_protocols
TRAIN_FLAC="$ASVSPOOF_ROOT/LA/ASVspoof2019_LA_train/flac"

if [ -d "$TRAIN_FLAC" ] && [ "$(ls -1 "$TRAIN_FLAC" | head -1)" ]; then
  echo "== Already extracted at $TRAIN_FLAC, skipping unzip =="
else
  echo "== Extracting into $ASVSPOOF_ROOT (121,461 files, several minutes) =="
  unzip -q -o "$ZIP" -d "$ASVSPOOF_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------
echo "== Layout =="
for d in ASVspoof2019_LA_train/flac ASVspoof2019_LA_dev/flac ASVspoof2019_LA_eval/flac; do
  p="$ASVSPOOF_ROOT/LA/$d"
  if [ -d "$p" ]; then
    printf '  %-34s %s files\n' "$d" "$(ls -1 "$p" | wc -l)"
  else
    printf '  %-34s MISSING\n' "$d"
  fi
done
echo "  protocols:"
ls -1 "$ASVSPOOF_ROOT/LA/ASVspoof2019_LA_cm_protocols" 2>/dev/null | sed 's/^/    /'

echo "== Disk after =="
df -h "$SCRATCH_DIR" | tail -1
echo
echo "LA.zip is kept at $ZIP so a re-extract needs no re-download."
echo "Delete it to reclaim $(numfmt --to=iec $LA_BYTES) once training runs clean."
echo
echo "Next: sbatch hpc/train.slurm"
