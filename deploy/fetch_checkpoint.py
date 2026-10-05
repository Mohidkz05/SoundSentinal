"""Downloads the served checkpoint into $CKPT_ROOT before the server starts.

Hosted (deploy/azure.sh), the weights live in a private blob container and the
container gets a read-only SAS for it: $CKPT_BASE_URL is the container URL,
$CKPT_SAS the query string. Container Apps express can't mount Azure Files, so
each cold start pulls them into the container's own filesystem.

Unset $CKPT_BASE_URL (local runs) and this does nothing.
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# Only what app.py reads: the checkpoint and its measured error rates.
FILES = ["best.pth"] + [f"best.measured-{s}.json" for s in ("itw", "la", "speechfake", "synth")]


def main():
    base = os.getenv("CKPT_BASE_URL")
    if not base:
        return
    sas = os.environ["CKPT_SAS"].lstrip("?")
    root = Path(os.environ["CKPT_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = root / name
        if dest.exists():
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        # The URL carries the SAS; never print it.
        with urllib.request.urlopen(f"{base.rstrip('/')}/{name}?{sas}", timeout=60) as r, \
                open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=8 * 1024 * 1024)
        part.rename(dest)
        print(f"fetched {name} ({dest.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
