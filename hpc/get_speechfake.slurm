#!/bin/bash
# Download and extract SpeechFake's bilingual subset and its real speech into scratch.
#
#   sbatch hpc/get_speechfake.slurm
#   sbatch --export=ALL,WITH_COMMONVOICE=1 hpc/get_speechfake.slurm   # + Common Voice
#
# WHY. SSL-AASIST + RawBoost trained on LA alone scores 11.21% EER on
# In-the-Wild (Finding 8). A 2026 comparison of anti-spoofing training sets
# (arXiv 2606.08038) found the number of generation methods matters more than
# hours: SpeechFake's bilingual subset (859 h, 30 methods) reached 2.63% on
# In-the-Wild against 17.20% for ASVspoof 5 — consistent with our own Finding 7,
# where adding ASVspoof 5 did not help.
#
# WHY THIS ONE. Chosen 26 September 2026:
#   - SpoofCeleb (3.57% there): CC BY 4.0, but gated behind manual approval
#     with an institutional email.
#   - AUDETER (claims 1.87%): partly generated FROM In-the-Wild's bona fide
#     audio. Training on it leaks our test set, so it is excluded outright.
#   - MLAAD: CC BY-NC 4.0, i.e. no commercial use.
#   SpeechFake is CC BY 4.0 and ungated. BD's real speech is VCTK, LibriTTS
#   (CC BY 4.0) and AISHELL-1/3 (Apache 2.0): read speech, mostly clean, so
#   this adds generator diversity but not In-the-Wild's noisy recording
#   conditions. Its 10 API-generated systems are withheld by the authors for
#   licensing reasons, so the release has 30.
#
# WHAT IS FETCHED. BD (bilingual English/Chinese, a 7-part split zip, ~275 GB)
# and the four Real/ zips BD's protocols reference (~13 GB: VCTK, LibriTTS,
# AISHELL-1/3), plus metadata and the licence. Real/CommonVoice.zip (20 GB) is
# NOT fetched by default: no row of the baseline protocol points into it — it
# is the bona fide half of MD. WITH_COMMONVOICE=1 adds it: 167,572 real clips
# recorded by volunteers on their own microphones (50,000 English), CC0, and
# unseen in training. It is the CALIBRATION set for the decision threshold
# (ai_model/calibrate.py; RESULTS.md Finding 9), never training data. MD (multilingual,
# ~115 GB) is NOT fetched: the result above is for BD. Add it to PARTS if the
# cross-lingual rows are ever wanted.
#
# In-the-Wild STAYS evaluation-only; nothing here touches it.
#
# Every file is checked against the SHA-256 ModelScope publishes for it. Re-running
# resumes: a verified file is skipped, a partial one is continued with curl -C -,
# and an extracted archive has a marker.
#
# SPEED. A fresh connection gets 50-90 MB/s from ModelScope's CDN (measured 26
# September 2026, login and compute nodes alike), but a long-lived one decays
# to ~2 MB/s — the first version of this job crawled at 4 MB/s in total. So
# curl aborts any transfer under 5 MB/s for 60 s and the loop resumes it on a
# fresh connection. At the fresh-connection rate the download is ~1.5 h. At
# busy times (evening AEST) connecting alone can take 10+ s and some attempts
# never start; the loop just keeps reopening.
#
#SBATCH --job-name=speechfake-download
#SBATCH --account=df37
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env.sh

API="https://www.modelscope.cn/api/v1/datasets/inclusionAI/SPEECHFAKE/repo?Revision=master&FilePath="
ROOT="${SPEECHFAKE_ROOT:-$SCRATCH_DIR/speechfake}"
ZIPS="$ROOT/zips"
mkdir -p "$ZIPS"

# path size sha256 — from ModelScope's repo tree API, 25 September 2026.
PARTS="
LICENSE.txt 2015 72d86d64f9c8015cd6021bd4d457084e4a010403aa73f1dc62b9d340f7efacad
metadata/metadata.zip 44765879 3e71dec782599b172d5181322b1a30e58c68f862b24b5f686987625ebd4a78d8
Real/VCTK-Corpus.zip 2498726226 db1f7f01973f00ea612cd2b110d06b3fabbc504d2fdd69646b66beed09eef0b9
Real/LibriTTS.zip 4597020747 60aeb64f973957127b096508aff29dc756be50c39ff1b509d8dabdd703abd736
Real/Aishell.zip 3360424492 eb8529440ad09fea0ad2b8abc4a30446745b93c6e16e396473d7b5a211969068
Real/Aishell3.zip 2571967588 376ffcd322a74298bb364bd7e42b889081d9cbb40078308925f0cb2e9c192302
BD/BD.z01 42949672960 0c49f08578bfb6f359dc8a2758e18656b3094ff133dc1d24fb77394629261fe6
BD/BD.z02 42949672960 489f62b3df6d1e04a9481af22e415301db7c2e7101be5ac98d1af360a9efc8c2
BD/BD.z03 42949672960 d60f13921187c5cdf33eaa4bae33c686c83b5caa9b71b645496eaf8e100276ab
BD/BD.z04 42949672960 55552e42de34ac933cfb59ae471571b075d4fd3d83260ba7fcd07c42c1780ecc
BD/BD.z05 42949672960 a1d1629ddbbf7f1ffda0dd93406064d5dbb99a62640334d5edf5fd5957de7a97
BD/BD.z06 42949672960 2198a18a00bdc6a29995ce3d57d38c8171b38324bc43cb74017b32377ce068a4
BD/BD.zip 17754620592 36907ce9058b062a88c5786216b3cabcae5f744fca7da0394d30fcae94d64208
"
if [ "${WITH_COMMONVOICE:-0}" = 1 ]; then
  PARTS="$PARTS
Real/CommonVoice.zip 20614616673 987f93a92f6a1dde4686604a114e8ccb7cdea8af461638269643272cd77bf2c1
"
fi

echo "== Disk before =="
df -h "$SCRATCH_DIR" | tail -1

if ! curl -sS -I --max-time 30 -L "${API}LICENSE.txt" >/dev/null 2>&1; then
  echo "ERROR: cannot reach modelscope.cn from this node." >&2
  echo "       See 'No internet' in hpc/README.md." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Download and verify
# ---------------------------------------------------------------------------
# The API answers with a 302 to a CDN URL carrying an expiring auth_key, so
# every attempt goes back through the API rather than reusing a CDN link. The
# CDN honours Range (checked: 206), which is what makes -C - resume work. As
# with get_asvspoof5.slurm: a short file means "resume", never "discard".
fetch() {
  local path=$1 want=$2 sha=$3 f="$ZIPS/$1" have
  mkdir -p "$(dirname "$f")"
  [ -f "$f.ok" ] && { echo "== $path: verified earlier, skipping =="; return 0; }
  for attempt in $(seq 1 1000); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && break
    [ "$have" -gt "$want" ] && { echo "   $path larger than expected; restarting"; rm -f "$f"; }
    # -Y/-y: give up on a connection that has decayed (see SPEED) and let the
    # loop reopen it. Exit 28 is that abort, so it resumes without sleeping.
    # NO --retry: curl's own retry restarts from the offset that invocation
    # began at and truncates the file back to it, throwing away everything
    # downloaded since (seen 26 September: 788 MB fell back to 561 MB). This
    # loop re-reads the file size each time, so it is the only retry.
    curl -sSL -C - -Y 5000000 -y 60 --connect-timeout 60 \
         -o "$f" "${API}${path}" 2>/dev/null || { [ $? -eq 28 ] || sleep 30; }
  done
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$have" -eq "$want" ] || { echo "ERROR: $path stuck at $have of $want bytes" >&2; return 1; }
  if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then
    touch "$f.ok"; echo "== $path: $(du -h "$f" | cut -f1), sha256 ok =="
  else
    echo "ERROR: $path sha256 mismatch on a complete file; deleted, re-submit to refetch" >&2
    rm -f "$f"; return 1
  fi
}

# Four at a time, as a pool: a slot is refilled the moment ANY download ends.
# (The first version waited on the oldest, so a slow file at the head of the
# queue held finished slots empty and only two ran.) Fourteen at once would
# mostly contend for the same link and the same scratch disk.
failed=0
running=0
while read -r path size sha; do
  [ -z "$path" ] && continue
  if [ "$running" -ge 4 ]; then
    wait -n || failed=1
    running=$((running - 1))
  fi
  fetch "$path" "$size" "$sha" &
  running=$((running + 1))
done <<< "$PARTS"
while [ "$running" -gt 0 ]; do
  wait -n || failed=1
  running=$((running - 1))
done
if [ "$failed" -ne 0 ]; then
  echo "ERROR: at least one file did not verify. Re-submit; verified files are kept." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Extract
# ---------------------------------------------------------------------------
# BD is a split zip (BD.z01..z06 + BD.zip). Info-ZIP's unzip cannot read split
# archives; 7-Zip can, given the .zip, and finds the .zNN parts beside it.
set +u                                  # module scripts read unset variables
module load 7zip/23.01 >/dev/null 2>&1 || true
set -u
command -v 7zz >/dev/null || { echo "ERROR: 7zz not found after 'module load 7zip/23.01'" >&2; exit 1; }
extract() {
  local zip=$1 dest=$2 marker="$ROOT/.$(basename "$1").extracted"
  [ -f "$marker" ] && { echo "== $(basename "$zip") already extracted =="; return 0; }
  echo "== Extracting $(basename "$zip") =="
  mkdir -p "$dest"
  7zz x -y -bso0 -bsp0 -o"$dest" "$zip"
  touch "$marker"
}
extract "$ZIPS/metadata/metadata.zip" "$ROOT"
for z in "$ZIPS"/Real/*.zip; do extract "$z" "$ROOT/Real"; done
extract "$ZIPS/BD/BD.zip" "$ROOT"
cp "$ZIPS/LICENSE.txt" "$ROOT/"

# ---------------------------------------------------------------------------
# 3. Report what is actually there
# ---------------------------------------------------------------------------
echo
echo "== Layout of $ROOT (two levels) =="
find "$ROOT" -mindepth 1 -maxdepth 2 -type d -not -path "$ZIPS*" | sort | while read -r d; do
  printf '  %-60s %s entries\n' "${d#$ROOT/}" "$(ls -1 "$d" | wc -l)"
done
echo
echo "== Audio files by top-level directory =="
for d in "$ROOT"/*/; do
  [ "$d" = "$ZIPS/" ] && continue
  printf '  %-20s %s\n' "$(basename "$d")" \
    "$(find "$d" -type f \( -name '*.wav' -o -name '*.flac' -o -name '*.mp3' \) | wc -l)"
done
echo
echo "== Baseline protocol files =="
find "$ROOT/metadata" -path '*baseline*' -name '*.csv' | sort | while read -r f; do
  echo "--- ${f#$ROOT/} ($(wc -l < "$f") lines) ---"
  head -3 "$f"
done

echo
echo "== Disk after =="
df -h "$SCRATCH_DIR" | tail -1
echo
echo "The zips are kept in $ZIPS; delete them once the loader has been checked"
echo "against the protocol files, to reclaim ~310 GB."
