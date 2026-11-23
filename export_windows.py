#!/usr/bin/env python3
"""Export a trained profile (.npz) to plain JSON for live_windows.py, which runs
on a stock Windows Python with no NumPy.

  uv run python export_windows.py profiles/<device>.npz [--thr 0.7 --hold-ms 100]
"""
import argparse, json, os

import gru_np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--hold-ms", type=float, default=100)
    ap.add_argument("--out")
    args = ap.parse_args()
    p = gru_np.load(args.profile)
    name = os.path.splitext(os.path.basename(args.profile))[0]
    out = args.out or os.path.join(HERE, "windows", "profiles", name + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = {"source": os.path.basename(args.profile), "trained_at": os.path.getmtime(args.profile),
           "thr": args.thr, "hold_ms": args.hold_ms,
           **{k: p[k].tolist() for k in gru_np.NAMES}}
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
