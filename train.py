#!/usr/bin/env python3
"""Train a tiny streaming GRU that flags wheel reports as jitter, one report at a time.

Split: per mode, the earliest session trains and the later ones validate
(leave-session-out, no window leakage). The model is causal: the decision for a
report uses only that report and the ones before it, so it adds zero latency.
"""
import argparse, json, os, random

import numpy as np
import torch
from torch import nn

import dataset
from baseline import run_filter

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, BURN_IN = 64, 16


class JitterGRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.head(out).squeeze(-1), h


def split(sessions):
    by_mode = {}
    for s in sessions:
        by_mode.setdefault(s["mode"], []).append(s)
    train = [group[0] for group in by_mode.values()]
    val = [s for group in by_mode.values() for s in group[1:]]
    return train, val


def windows(sessions):
    xs, ys, ws = [], [], []
    for s in sessions:
        f, lab, notch, first = s["feats"], s["labels"], s["notches"], s["first_labelled"]
        for start in range(0, len(f) - WINDOW + 1, 4):
            xs.append(f[start:start + WINDOW])
            ys.append(lab[start:start + WINDOW])
            mask = [0.0 if (i < BURN_IN or start + i < first) else float(notch[start + i])
                    for i in range(WINDOW)]
            ws.append(mask)
    return torch.tensor(xs), torch.tensor(ys, dtype=torch.float32), torch.tensor(ws)


@torch.no_grad()
def stream_probs(model, s):
    logits, _ = model(torch.tensor([s["feats"]]))
    return torch.sigmoid(logits[0]).numpy()


def score(suppress, s):
    """Notch-weighted: share of jitter suppressed, share of good input dropped."""
    first = s["first_labelled"]
    lab = np.array(s["labels"][first:])
    notch = np.array(s["notches"][first:])
    sup = np.array(suppress[first:], dtype=bool)
    jit = notch[lab == 1].sum()
    good = notch[lab == 0].sum()
    return notch[(lab == 1) & sup].sum(), jit, notch[(lab == 0) & sup].sum(), good


def baseline_suppress(s, M=4, W=80):
    """The rule filter as a per-report suppress mask (reports it never emitted)."""
    meta, ticks = dataset.load(os.path.join(dataset.SESSIONS, s["name"]))
    emitted = {src for _, _, src in run_filter(ticks, M, W)}
    return [t not in emitted for t, _ in ticks]


def report(label, rows):
    js = sum(r[0] for r in rows); jt = sum(r[1] for r in rows)
    gs = sum(r[2] for r in rows); gt = sum(r[3] for r in rows)
    print(f"  {label:<22} jitter suppressed {js:4d}/{jt:4d} ({100 * js / jt:5.1f}%)   "
          f"good dropped {gs:4d}/{gt:5d} ({100 * gs / gt:4.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    train, val = split(dataset.all_sessions())
    print("train:", [s["name"] for s in train])
    print("val:  ", [s["name"] for s in val])
    x, y, w = windows(train)
    model = JitterGRU(args.hidden)
    params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU hidden={args.hidden}, {params} params, {len(x)} training windows")

    pos_weight = float((w * (1 - y)).sum() / (w * y).sum())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            logits, _ = model(x[idx])
            loss = (lossf(logits, y[idx]) * w[idx]).sum() / w[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 9:
            print(f"  epoch {epoch + 1:3d} loss {total / len(x):.4f}")

    print("\nvalidation (unseen sessions, notch-weighted):")
    report("rule baseline M=4 W=80", [score(baseline_suppress(s), s) for s in val])
    probs = {s["name"]: stream_probs(model, s) for s in val}
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        report(f"GRU threshold {thr}", [score(probs[s["name"]] > thr, s) for s in val])
    print("\nper session at threshold 0.8:")
    for s in val:
        report(s["name"], [score(probs[s["name"]] > 0.8, s)])

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    out = os.path.join(HERE, "models", f"gru_h{args.hidden}.pt")
    torch.save({"state": model.state_dict(), "hidden": args.hidden,
                "train": [s["name"] for s in train]}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
