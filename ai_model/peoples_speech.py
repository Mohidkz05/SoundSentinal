# peoples_speech.py
#
# The People's Speech (MLCommons, NeurIPS 2021 Datasets track): English
# speech from archive.org — talks, lectures, public meetings, podcasts,
# government proceedings. Only the `clean` configuration is used, which is
# CC-BY (2.0-4.0): commercial use allowed, with attribution. The `_sa`
# configurations are CC-BY-SA and are deliberately not fetched, so that no
# share-alike term can reach a model artifact.
#
# WHY IT IS HERE. It is the CALIBRATION set for Finding 11, and is never
# training data. Finding 10 showed that a threshold fitted on a source the
# model trained on is useless: the model learns that source's recording
# conditions, its scores collapse into a narrow band, and the threshold lands
# inside it. So the threshold must come from a source held out of training
# entirely. People's Speech is also, unlike Common Voice and VoxPopuli, not
# in XLS-R's pretraining data, so its audio is unfamiliar to the front-end
# too.
#
# WHICH CLIPS. `clean` test split, 34,898 clips, 16 kHz FLAC. There are no
# speaker ids. Each id begins with its source recording's name, which is used
# as `speaker_id` so that calibrate.py assigns whole recordings to one half:
# the check half is then recordings the threshold was not fitted to.
#
# IN-THE-WILD SPEAKERS. In-the-Wild's genuine clips are public figures, and
# archive.org holds speeches by some of them. A recording of, say, a
# presidential address would pull the calibration set towards the test set.
# ITW_PATTERNS lists a name pattern for every one of In-the-Wild's 54
# speakers (from its meta.csv, 28 September 2026). Any recording whose name
# matches is dropped before sampling. The list was fixed before any
# calibration was run. It matches recording names only; a speaker nobody
# named in the title is not caught.
#
# In-the-Wild stays evaluation-only. Nothing here reads its audio.
#
#     python peoples_speech.py extract <dir with the .parquet shards> <out dir>
#
# writes one FLAC per clip and a clips.csv (file, speaker_id, duration_ms,
# itw_match). Fetch the shards with hpc/get_peoples_speech.slurm.

import os
import re
import sys
from pathlib import Path

import pandas as pd

CLIPS_CSV = "clips.csv"

# One regex per In-the-Wild speaker, matched as whole words against the
# recording name after camelCase, _, -, . and "_DOT_" are turned into spaces.
# Distinctive surnames stand alone; common ones need the full name.
ITW_PATTERNS = {
    "2Pac": r"2pac|tupac",
    "Adam Driver": r"adam driver",
    "Alan Watts": r"alan watts",
    "Alec Guinness": r"guinness",
    "Alexandria Ocasio-Cortez": r"ocasio|aoc",
    "Arnold Schwarzenegger": r"schwarzenegger",
    "Ayn Rand": r"ayn rand",
    "Barack Obama": r"obama",
    "Bernie Sanders": r"bernie",
    "Bill Burr": r"bill burr",
    "Bill Clinton": r"clinton",
    "Billie Eilish": r"eilish",
    "Bob Ross": r"bob ross",
    "Boris Johnson": r"boris",
    "Calvin Coolidge": r"coolidge",
    "Christopher Hitchens": r"hitchens",
    "Dave Chappelle": r"chappelle",
    "Donald Trump": r"trump",
    "Dwight Eisenhower": r"eisenhower",
    "FDR": r"fdr|roosevelt",
    "Frank Sinatra": r"sinatra",
    "George Carlin": r"carlin",
    "George W. Bush": r"bush",
    "Gilbert Gottfried": r"gottfried",
    "Harry Truman": r"truman",
    "JFK": r"jfk|kennedy",
    "Jeff Goldblum": r"goldblum",
    "Jerry Seinfeld": r"seinfeld",
    "Jimmy Carter": r"jimmy carter",
    "John Cleese": r"cleese",
    "Kamala Harris": r"kamala",
    "Kanye West": r"kanye",
    "Louis C.K.": r"louis c ?k",
    "Louis Farrakhan": r"farrakhan",
    "Lyndon Johnson": r"lyndon|lbj",
    "Malcolm X": r"malcolm x",
    "Mark Zuckerberg": r"zuckerberg",
    "Martin Luther King": r"luther king|mlk",
    "Milton Friedman": r"milton friedman",
    "Mitch Hedberg": r"hedberg",
    "Mr. Rogers": r"mr rogers|mister rogers|fred rogers",
    "Nelson Mandela": r"mandela",
    "Nick Offerman": r"offerman",
    "Norm MacDonald": r"norm macdonald",
    "Orson Welles": r"welles",
    "Queen Elizabeth II": r"queen elizabeth",
    "Richard Nixon": r"nixon",
    "Robert Kardashian": r"kardashian",
    "Ronald Reagan": r"reagan",
    "Scarlett Johansson": r"johansson",
    "The Notorious B.I.G.": r"notorious|biggie",
    "Tucker Carlson": r"tucker carlson",
    "William F. Buckley": r"buckley",
    "Winston Churchill": r"churchill",
}
_ITW_RE = re.compile(r"\b(" + "|".join(ITW_PATTERNS.values()) + r")\b")


def normalise(name):
    """Recording name -> lowercase words, for ITW_PATTERNS."""
    name = name.replace("_DOT_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def itw_match(recording):
    """The In-the-Wild pattern a recording name matches, or ""."""
    m = _ITW_RE.search(normalise(recording))
    return m.group(0) if m else ""


def get_peoples_speech_root():
    env = os.getenv("PEOPLES_SPEECH_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "peoples_speech"
    if not (root / CLIPS_CSV).is_file():
        raise FileNotFoundError(
            f"No {CLIPS_CSV} under {root}.\n"
            f"Fetch and extract it first:  sbatch hpc/get_peoples_speech.slurm\n"
            f"$PEOPLES_SPEECH_ROOT is currently {env or '(unset)'}.")
    return root


def load_clips(root=None):
    """(frame with file + speaker_id, root), In-the-Wild matches already dropped."""
    root = Path(root) if root is not None else get_peoples_speech_root()
    clips = pd.read_csv(root / CLIPS_CSV, dtype=str, keep_default_na=False)
    return clips[clips["itw_match"] == ""].reset_index(drop=True), root


def extract(shard_dir, out_dir):
    """Unpack the FLAC bytes embedded in the Parquet shards into files, unchanged."""
    import pyarrow.parquet as pq

    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    rows = []
    for shard in sorted(shard_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=256, columns=["id", "audio", "duration_ms"]):
            for rec in batch.to_pylist():
                # ids look like <recording>/<recording>/<recording>_DOT_flac_00066.flac
                recording = rec["id"].split("/")[0]
                rel = Path("audio") / rec["id"]
                target = out_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not (target.is_file() and target.stat().st_size == len(rec["audio"]["bytes"])):
                    target.write_bytes(rec["audio"]["bytes"])
                rows.append({"file": str(rel), "speaker_id": recording,
                             "duration_ms": rec["duration_ms"],
                             "itw_match": itw_match(recording)})
        print(f"  {shard.name}: done, {len(rows)} clips so far", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / CLIPS_CSV, index=False)
    hits = frame[frame["itw_match"] != ""]
    print(f"Wrote {len(frame)} clips from {frame['speaker_id'].nunique()} recordings "
          f"to {out_dir / CLIPS_CSV}")
    print(f"In-the-Wild name matches: {hits['speaker_id'].nunique()} recordings, "
          f"{len(hits)} clips, excluded from load_clips()")
    for rec, pat in hits.groupby("speaker_id")["itw_match"].first().items():
        print(f"  {pat!r:16} {rec}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        raise SystemExit("usage: python peoples_speech.py extract <shard dir> <out dir>")
    extract(sys.argv[2], sys.argv[3])
