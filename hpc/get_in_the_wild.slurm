#!/bin/bash
# Download and extract the In-the-Wild dataset into scratch.
#
#   sbatch hpc/get_in_the_wild.slurm
#
# WHAT THIS IS FOR. In-the-Wild (Müller et al., 2022) holds 37.9 hours of audio
# — 20.7 real, 17.2 fake — for 58 celebrities and politicians, collected from
# social networks and video platforms. Unlike ASVspoof2019 these are deepfakes
# people actually made and posted, not attacks generated in a lab to a protocol.
#
# It is an EVALUATION SET ONLY. It is never trained on and never selected on.
# Our LA numbers stay comparable to published work precisely because the
# training set does not change; this measures whether any of it generalises.
# ASVspoof2019's attacks predate neural codec models and current commercial
# voice cloning, so a model at 3% EER on LA eval may be far worse on a clip a
# real user would upload — and that gap is a result worth reporting.
#
# Expect a much worse number than LA eval. The literature reports detectors
# falling from around 1% EER on ASVspoof2019 to roughly 30-40% here. If ours
# lands in that range it is reproducing a known finding, not a broken pipeline.
#
# Submitted as a job rather than run at the login prompt for the same reason
# get_la.slurm is: 8.2GB in, ~32k WAVs out, and login nodes are for
# "light-weight account housekeeping". No GPU — network and filesystem bound.
#
# Re-running is safe and is the intended way to resume: curl -C - continues a
# partial file, and both stages are skipped if their output is already complete.
#
#SBATCH --job-name=itw-download
#SBATCH --account=df37
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

# Always the huggingface.co URL, NEVER the CDN address it redirects to. That
# redirect is a signed AWS URL carrying an Expires timestamp and a Signature;
# copying it out of a browser or a previous log gives a link that works for a
# few hours and then serves 403 forever. curl -L re-resolves it each run.
#
# Verified 21 September 2026: Content-Type application/zip, Accept-Ranges
# bytes (which is what makes the resume below work), 8161489023 bytes.
ITW_URL="https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
ITW_BYTES=8161489023

ZIP="$SCRATCH_DIR/in_the_wild.zip"
mkdir -p "$SCRATCH_DIR" "$ITW_ROOT"

echo "== Disk before =="
df -h "$SCRATCH_DIR" | tail -1

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
have=0
[ -f "$ZIP" ] && have=$(stat -c %s "$ZIP")

if [ "$have" -eq "$ITW_BYTES" ]; then
  echo "== in_the_wild.zip already complete ($ITW_BYTES bytes), skipping download =="
else
  if ! curl -sS -I --max-time 30 -L "$ITW_URL" >/dev/null 2>&1; then
    echo "ERROR: cannot reach huggingface.co from this node." >&2
    echo "       Compute nodes may have no outbound route. Run the download on" >&2
    echo "       the data transfer node instead — see 'No internet' in hpc/README.md." >&2
    exit 1
  fi
  echo "== Downloading in_the_wild.zip ($(numfmt --to=iec $ITW_BYTES)) =="
  [ "$have" -gt 0 ] && echo "   resuming from $(numfmt --to=iec "$have")"
  echo "   (no progress meter by design — watch the file grow instead:"
  echo "    ssh m3 'ls -lh $ZIP')"
  curl -L -C - --no-progress-meter --retry 10 --retry-delay 15 --retry-connrefused \
       -o "$ZIP" "$ITW_URL"

  got=$(stat -c %s "$ZIP")
  if [ "$got" -ne "$ITW_BYTES" ]; then
    echo "ERROR: in_the_wild.zip is $got bytes, expected $ITW_BYTES." >&2
    echo "       Re-submit to resume; delete the file first if it is HTML." >&2
    exit 1
  fi
  echo "   size verified: $got bytes"
fi

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
# The archive's internal layout is NOT documented on the dataset card, so this
# script does not assume one — it extracts into $ITW_ROOT and then reports what
# actually landed. The evaluation adapter is written against that report rather
# than against a guess. If a later release changes the layout, the report below
# is what tells you, instead of a confusing failure inside the dataloader.
if [ -n "$(ls -A "$ITW_ROOT" 2>/dev/null)" ]; then
  echo "== $ITW_ROOT is not empty, skipping unzip =="
else
  echo "== Extracting into $ITW_ROOT (~32k files, several minutes) =="
  unzip -q -o "$ZIP" -d "$ITW_ROOT"
fi

# ---------------------------------------------------------------------------
# 3. Report what is actually there
# ---------------------------------------------------------------------------
echo
echo "== Top level of $ITW_ROOT =="
ls -1 "$ITW_ROOT" | head -20
echo
echo "== Directory tree, two levels, with file counts =="
find "$ITW_ROOT" -maxdepth 2 -type d | while read -r d; do
  printf '  %-60s %s entries\n' "${d#$ITW_ROOT/}" "$(ls -1 "$d" 2>/dev/null | wc -l)"
done
echo
echo "== Audio file extensions present =="
find "$ITW_ROOT" -type f ! -name '*.csv' ! -name '*.txt' ! -name '*.md' \
  | sed 's/.*\.//' | sort | uniq -c | sort -rn | head
echo
echo "== Metadata files, with header and first rows =="
find "$ITW_ROOT" -maxdepth 2 -type f \( -name '*.csv' -o -name '*.txt' -o -name '*.md' \) | while read -r f; do
  echo "--- ${f#$ITW_ROOT/} ($(wc -l < "$f") lines) ---"
  head -5 "$f"
  echo
done
echo "== Label values and counts (if a meta csv exists) =="
META=$(find "$ITW_ROOT" -maxdepth 2 -name 'meta*.csv' | head -1)
if [ -n "$META" ]; then
  awk -F, 'NR>1 {print $NF}' "$META" | sort | uniq -c
else
  echo "  no meta*.csv found — see the metadata listing above"
fi

echo
echo "== Disk after =="
df -h "$SCRATCH_DIR" | tail -1
echo
echo "in_the_wild.zip is kept at $ZIP so a re-extract needs no re-download."
echo
echo "Next: the evaluation adapter is written against the layout reported above."
