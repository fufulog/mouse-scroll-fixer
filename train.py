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

import dataset, gru_np
from baseline import run_filter

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, BURN_IN = 64, 16


class JitterGRU(nn.Module):
    def __init__(self, hidden, in_dim=dataset.IN_DIM):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
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


def windows(sessions, syn_frac=0.15, seed=0):
    xs, ys, ws = [], [], []
    for s in sessions:
        f, lab, notch, first = s["feats"], s["labels"], s["notches"], s["first_labelled"]
        for start in range(0, len(f) - WINDOW + 1, 4):
            xs.append(f[start:start + WINDOW])
            ys.append(lab[start:start + WINDOW])
            mask = [0.0 if (i < BURN_IN or start + i < first) else float(notch[start + i])
                    for i in range(WINDOW)]
            ws.append(mask)
    if syn_frac > 0:
        sx, sy, sw = dataset.synthetic_reversals(sessions, int(len(xs) * syn_frac), WINDOW, BURN_IN, seed=seed)
        xs += sx
        ys += sy
        ws += sw
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32), torch.tensor(ws, dtype=torch.float32)


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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--syn-frac", type=float, default=0.15)
    ap.add_argument("--all", action="store_true", help="train on all labelled sessions (no val split)")
    ap.add_argument("--export-base", action="store_true", help="export weights/base.npz")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    sessions = dataset.all_sessions()
    if args.all:
        train, val = sessions, []
    else:
        train, val = split(sessions)
    print("train:", [s["name"] for s in train])
    if val:
        print("val:  ", [s["name"] for s in val])
    x, y, w = windows(train, syn_frac=args.syn_frac, seed=args.seed)
    model = JitterGRU(args.hidden, dataset.IN_DIM)
    params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU in={dataset.IN_DIM} hidden={args.hidden}, {params} params, {len(x)} training windows")

    pos_weight = float((w * (1 - y)).sum() / (w * y).sum())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight))
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            logits, _ = model(x[idx])
            loss = (lossf(logits, y[idx]) * w[idx]).sum() / w[idx].sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if epoch % 10 == 9:
            print(f"  epoch {epoch + 1:3d} loss {total / len(x):.4f}")

    if not args.all:
        print("\nvalidation (unseen sessions, notch-weighted):")
        report("rule baseline M=4 W=80", [score(baseline_suppress(s), s) for s in val])
        probs = {s["name"]: stream_probs(model, s) for s in val}
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            report(f"GRU threshold {thr}", [score(probs[s["name"]] > thr, s) for s in val])
        print("\nper session at threshold 0.7:")
        for s in val:
            report(s["name"], [score(probs[s["name"]] > 0.7, s)])

        os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
        out = os.path.join(HERE, "models", f"gru_h{args.hidden}.pt")
        torch.save({"state": model.state_dict(), "hidden": args.hidden, "in_dim": dataset.IN_DIM,
                    "train": [s["name"] for s in train]}, out)
        print(f"\nsaved {out}")

    if args.export_base:
        os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
        base_out = os.path.join(HERE, "weights", "base.npz")
        gru_np.save(base_out, gru_np.from_torch_state(model.state_dict()),
                    in_dim=dataset.IN_DIM, hidden=args.hidden, thr=0.7, hold_ms=100,
                    train=[s["name"] for s in train])
        print(f"exported {base_out}")


if __name__ == "__main__":
    main()
