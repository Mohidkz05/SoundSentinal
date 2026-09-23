#!/bin/bash
# Download and extract the ASVspoof 5 train and dev partitions into scratch.
#
#   sbatch hpc/get_asvspoof5.slurm
#
# WHY. AASIST trained on ASVspoof2019 LA scores 3.17% EER on LA eval and
# 37.15% on In-the-Wild (Finding 6 in RESULTS.md). LA's attacks predate modern
# TTS and voice conversion, and its bona fide audio is one clean studio
# corpus (VCTK). ASVspoof 5 (2024) fixes both: 2,000+ crowdsourced speakers
# recorded in diverse conditions (from Multilingual LibriSpeech), and newer
# TTS/VC attacks. It is the candidate for adding training data.
#
# WHY THIS ONE AND NOT MLAAD. ASVspoof 5 is ODC-By with CC BY 4.0 bona fide
# audio — the same terms as ASVspoof2019, so it adds no new obligation. MLAAD
# is CC BY-NC 4.0, spoof-only (bona fide must come from M-AILABS separately)
# and 183 GB. Checked 23 September 2026 against the Zenodo record and the
# dataset card.
#
# In-the-Wild STAYS evaluation-only. Adding ASVspoof 5 to training is a new
# row in the table, not a change to the LA rows; the ITW number for such a
# model is legitimate precisely because neither corpus touches ITW.
#
# WHAT IS FETCHED. Train (5 tars, ~37.5 GB, 182,357 clips) and dev (3 tars,
# ~20 GB, 142,134 clips) plus protocols. The eval partition (11 tars, ~94 GB)
# is not fetched — it is a test set, and is only worth its size if we decide
# to report ASVspoof 5 eval numbers. Add the E tars to PARTS for that.
#
# Every tar is checked against the md5 printed in the README before it is
# extracted. Re-running is safe and resumes: curl -C - continues a partial
# file, a tar whose md5 already matches is not re-downloaded, and a partition
# whose .extracted marker exists is not re-extracted.
#
#SBATCH --job-name=asv5-download
#SBATCH --account=df37
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

# Zenodo record 14498691, "ASVspoof 5 Database", verified 23 September 2026.
RECORD="https://zenodo.org/records/14498691/files"
ASV5_ROOT="${ASV5_ROOT:-$SCRATCH_DIR/asvspoof5}"
TARS="$ASV5_ROOT/tars"
mkdir -p "$TARS"

# name md5 — copied verbatim from the record's README.txt, section 2.
PARTS="
flac_T_aa.tar b0cc86b14826a7701b52aad4f53daf9c
flac_T_ab.tar d05be3f4be7a343fdbdd0ed29fdff2e1
flac_T_ac.tar 70d1ba4ad75a20aef3dae541fbd321e3
flac_T_ad.tar c9bb56af2cc410d98338d74babcb95c9
flac_T_ae.tar a3969816982f3e52300d6147ad796df7
flac_D_aa.tar df0be44957623991028cce59792beb17
flac_D_ab.tar 1e8cd685d89b64502692f1bcf1a13db3
flac_D_ac.tar 5e0031f08c30e4bdbf0c59f91b2d662b
ASVspoof5_protocols.tar.gz 865d0e894ea9f686f0f37e5ae3ae3616
"

echo "== Disk before =="
df -h "$SCRATCH_DIR" | tail -1

if ! curl -sS -I --max-time 30 -L "$RECORD/README.txt?download=1" >/dev/null 2>&1; then
  echo "ERROR: cannot reach zenodo.org from this node." >&2
  echo "       See 'No internet' in hpc/README.md." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Download and verify
# ---------------------------------------------------------------------------
while read -r name md5; do
  [ -z "$name" ] && continue
  f="$TARS/$name"
  if [ -f "$f.ok" ]; then
    echo "== $name: verified earlier, skipping =="
    continue
  fi
  echo "== $name =="
  for attempt in 1 2; do
    curl -L -C - --no-progress-meter --retry 10 --retry-delay 30 --retry-connrefused \
         -o "$f" "$RECORD/$name?download=1" || true
    got=$(md5sum "$f" | cut -d' ' -f1)
    if [ "$got" = "$md5" ]; then
      echo "   md5 ok ($(du -h "$f" | cut -f1))"
      touch "$f.ok"
      break
    fi
    # A resumed download onto a corrupt partial never converges; start over once.
    echo "   md5 MISMATCH ($got, expected $md5) — discarding and retrying"
    rm -f "$f"
    if [ "$attempt" -eq 2 ]; then
      echo "ERROR: $name failed verification twice." >&2
      exit 1
    fi
  done
done <<< "$PARTS"

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
# README: flac_T_*.tar -> ./flac_T, flac_D_*.tar -> ./flac_D.
for p in T D; do
  if [ -f "$ASV5_ROOT/flac_$p.extracted" ]; then
    echo "== flac_$p already extracted =="
    continue
  fi
  echo "== Extracting flac_$p =="
  for t in "$TARS"/flac_${p}_*.tar; do
    tar -xf "$t" -C "$ASV5_ROOT"
  done
  touch "$ASV5_ROOT/flac_$p.extracted"
done
tar -xzf "$TARS/ASVspoof5_protocols.tar.gz" -C "$ASV5_ROOT"

# ---------------------------------------------------------------------------
# 3. Report what is actually there
# ---------------------------------------------------------------------------
echo
echo "== Layout of $ASV5_ROOT =="
find "$ASV5_ROOT" -maxdepth 2 -type d | while read -r d; do
  printf '  %-50s %s entries\n' "${d#$ASV5_ROOT}" "$(ls -1 "$d" | wc -l)"
done
echo
echo "== Protocol files =="
find "$ASV5_ROOT" -maxdepth 2 \( -name '*.tsv' -o -name '*.csv' \) | while read -r f; do
  echo "--- ${f#$ASV5_ROOT/} ($(wc -l < "$f") lines) ---"
  head -3 "$f"
done
echo
echo "== Clip counts vs README (train 182357, dev 142134) =="
for p in T D; do
  printf '  flac_%s: %s\n' "$p" "$(find "$ASV5_ROOT/flac_$p" -name '*.flac' 2>/dev/null | wc -l)"
done
TRAIN_TSV=$(find "$ASV5_ROOT" -name 'ASVspoof5.train.tsv' | head -1)
if [ -n "$TRAIN_TSV" ]; then
  echo
  echo "== Train keys and codecs =="
  awk '{print $9}' "$TRAIN_TSV" | sort | uniq -c
  awk '{print $4}' "$TRAIN_TSV" | sort | uniq -c | head
fi

echo
echo "== Disk after =="
df -h "$SCRATCH_DIR" | tail -1
echo
echo "The tars are kept in $TARS; delete them once the extract is verified"
echo "to reclaim ~58 GB."
