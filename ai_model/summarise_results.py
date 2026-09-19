# summarise_results.py
#
# Tabulate the JSON files evaluate.py writes.
#
#     python summarise_results.py <dir-or-files>...
#     python summarise_results.py $CKPT_ROOT/nodp --sort epoch
#
# Exists so the comparison table in APPROACH.md is assembled from measurements
# rather than retyped from terminal scrollback. Once there are several
# front-ends times two privacy regimes times several architectures, hand-copying
# numbers between a log and a markdown table is a transcription error waiting to
# be quoted in a report.

import argparse
import json
from pathlib import Path


def load(paths):
    out = []
    for p in paths:
        files = sorted(p.glob("eval_*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                out.append((f, json.loads(f.read_text())))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  (skipped {f.name}: {type(e).__name__})")
    return out


def fmt(v, spec, dash="—"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description="Tabulate evaluate.py result files.")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--sort", default="eer",
                    choices=["eer", "tdcf", "epoch", "time"],
                    help="Column to order rows by. Default eer.")
    args = ap.parse_args()

    rows = load(args.paths)
    if not rows:
        print("No result files found.")
        return

    key = {
        "eer":   lambda r: r[1]["pooled"]["eer"],
        "tdcf":  lambda r: r[1]["pooled"].get("min_tdcf") or 9e9,
        "epoch": lambda r: r[1].get("epoch") or 0,
        "time":  lambda r: r[1].get("timestamp") or "",
    }[args.sort]
    rows.sort(key=key)

    hdr = f"{'run':<26} {'front-end':<10} {'regime':<12} {'ep':>3} {'dev EER':>8} {'eval EER':>9} {'min t-DCF':>10}"
    print(hdr)
    print("-" * len(hdr))
    for path, r in rows:
        p = r["pooled"]
        dev = r.get("dev", {}).get("eer")
        print(f"{path.stem[:26]:<26} "
              f"{r.get('frontend', 'logmel'):<10} "
              f"{r.get('regime', '?'):<12} "
              f"{str(r.get('epoch', '?')):>3} "
              f"{fmt(None if dev is None else dev * 100, '7.2f') + '%':>8} "
              f"{p['eer'] * 100:8.2f}% "
              f"{fmt(p.get('min_tdcf'), '10.4f')}")

    # The comparison that motivated this script: does the dev number, which is
    # what best.pth is selected by, actually track the eval number?
    withdev = [(r["dev"]["eer"], r["pooled"]["eer"]) for _, r in rows
               if r.get("dev", {}).get("eer") is not None]
    if len(withdev) > 2:
        import statistics
        devs = [d for d, _ in withdev]
        evals = [e for _, e in withdev]
        try:
            rho = statistics.correlation(devs, evals)
            print(f"\ndev/eval EER correlation across {len(withdev)} runs: {rho:+.3f}")
            if rho < 0.5:
                print("  Weak or negative: selecting best.pth by dev EER is not")
                print("  selecting the model that generalises. Say so in the writeup.")
        except statistics.StatisticsError:
            pass


if __name__ == "__main__":
    main()
