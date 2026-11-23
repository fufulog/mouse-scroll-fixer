#!/usr/bin/env python3
"""Fine-tune the jitter model to one mouse, NumPy only.

Split: every session of the device is cut in time; the first 75% of its
reports trains, the last 25% scores (streamed over the whole session so the
GRU state is warm). Starts from base weights, keeps the epoch with the lowest
held-out loss, and reports base vs tuned under the live hold/release policy.

  python finetune.py <device path> [--out profiles/x.npz]
"""
import argparse, copy, os

import numpy as np

import dataset, gru_np, policy

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "weights", "base.npz")
WINDOW, BURN_IN, STRIDE, BATCH = 64, 16, 4, 64
TRAIN_FRAC, THR, HOLD_MS = 0.75, 0.7, 100
SYN_FRAC, WD = 0.15, 1e-2


def device_sessions(device):
    return [s for s in dataset.all_sessions(include_model_tests=True)
            if s["device"] == device and s["include_in_train"]]


def cut_index(s):
    f = s["first_labelled"]
    return f + int((len(s["labels"]) - f) * TRAIN_FRAC)


def train_windows(sessions, syn_frac=SYN_FRAC, seed=0):
    xs, ys, ws = [], [], []
    pad = [0.0] * dataset.IN_DIM
    for s in sessions:
        end, first = cut_index(s), s["first_labelled"]
        for start in range(0, max(end - WINDOW, 0) + 1, STRIDE):
            idx = range(start, start + WINDOW)
            xs.append([s["feats"][i] if i < end else pad for i in idx])
            ys.append([s["labels"][i] if i < end else 0 for i in idx])
            ws.append([float(s["notches"][i]) if (BURN_IN <= i - start and first <= i < end) else 0.0
                       for i in idx])
    if syn_frac > 0:
        sx, sy, sw = dataset.synthetic_reversals(
            sessions, int(len(xs) * syn_frac), WINDOW, BURN_IN, seed=seed,
            end_indices={s["name"]: cut_index(s) for s in sessions},
        )
        xs += sx
        ys += sy
        ws += sw
    return np.array(xs), np.array(ys, dtype=float), np.array(ws)


def session_probs(params, s):
    logits, _ = gru_np.forward(params, np.array([s["feats"]]))
    return gru_np.sigmoid(logits[0])


def heldout_loss(params, sessions, pos_weight):
    total = wsum = 0.0
    for s in sessions:
        p = session_probs(params, s)[cut_index(s):]
        y = np.array(s["labels"][cut_index(s):], dtype=float)
        w = np.array(s["notches"][cut_index(s):], dtype=float)
        l = -(pos_weight * y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        total += float((l * w).sum())
        wsum += float(w.sum())
    return total / max(wsum, 1e-9)


def heldout_metrics(params, sessions):
    """Held-out part only, under the live hold/release policy (notch-weighted)."""
    jp = jt = gl = gt = 0
    delays, eaten = [], []
    for s in sessions:
        meta, ticks = dataset.load(os.path.join(dataset.SESSIONS, s["name"]))
        sup = session_probs(params, s) > THR
        emitted = {src: et for et, _, src in policy.hold_release(ticks, sup, HOLD_MS)}
        cut = cut_index(s)
        for i in range(cut, len(ticks)):
            n = abs(ticks[i][1])
            if s["labels"][i]:
                jt += n
                jp += n if i in emitted else 0
                continue
            gt += n
            if i in emitted:
                if emitted[i] > ticks[i][0]:
                    delays.append((emitted[i] - ticks[i][0]) / 1e6)
            else:
                gl += n
        if s["mode"] in ("sequence", "model_test"):
            intents = dataset.sequence_intents(meta, ticks)
            for i in range(max(cut, 1), len(ticks)):
                if intents[i] != intents[i - 1] and intents[i - 1] != 0:
                    seg_end = i
                    while seg_end < len(ticks) and intents[seg_end] == intents[i]:
                        seg_end += 1
                    j, lost = i, 0
                    while j < seg_end:
                        if ticks[j][1] * intents[i] > 0:
                            if j in emitted:
                                break
                            lost += abs(ticks[j][1])
                        j += 1
                    eaten.append(lost)
    return {"jitter_passed": jp, "jitter_total": jt, "good_lost": gl, "good_total": gt,
            "delayed": len(delays), "delay_max": max(delays, default=0.0),
            "reversals": len(eaten), "reversal_lost_max": max(eaten, default=0)}


def finetune(device, base_path=BASE, epochs=30, lr=1e-3, progress=None, seed=0):
    sessions = device_sessions(device)
    if not sessions:
        raise SystemExit(f"no labelled sessions for {device}")
    rng = np.random.default_rng(seed)
    base = gru_np.load(base_path)
    if base["w_ih"].shape[1] != dataset.IN_DIM:
        raise SystemExit(f"incompatible base weights {base_path}: expects {base['w_ih'].shape[1]} features, need {dataset.IN_DIM}")
    x, y, w = train_windows(sessions, seed=seed)
    pos_weight = float((w * (1 - y)).sum() / max((w * y).sum(), 1e-9))
    params = copy.deepcopy(base)
    opt = gru_np.Adam(params, lr, wd=WD)
    best, best_loss = copy.deepcopy(params), heldout_loss(params, sessions, pos_weight)
    base_loss = best_loss
    for epoch in range(epochs):
        perm = rng.permutation(len(x))
        train_loss = 0.0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            loss, grads = gru_np.loss_and_grads(params, x[idx], y[idx], w[idx], pos_weight)
            opt.step(params, grads)
            train_loss += loss * len(idx)
        val = heldout_loss(params, sessions, pos_weight)
        if val < best_loss:
            best, best_loss = copy.deepcopy(params), val
        if progress:
            progress(epoch + 1, epochs, train_loss / len(x), val, best_loss)
    return {"base": base, "tuned": best, "sessions": sessions, "windows": len(x),
            "base_loss": base_loss, "tuned_loss": best_loss,
            "base_metrics": heldout_metrics(base, sessions),
            "tuned_metrics": heldout_metrics(best, sessions)}


def format_metrics(m):
    jp = 100 * m["jitter_passed"] / max(m["jitter_total"], 1)
    gl = 100 * m["good_lost"] / max(m["good_total"], 1)
    return (f"jitter passed {m['jitter_passed']}/{m['jitter_total']} ({jp:.1f}%)  "
            f"good lost {m['good_lost']}/{m['good_total']} ({gl:.2f}%)  "
            f"worst reversal: {m['reversal_lost_max']} notches lost over {m['reversals']} reversals")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out")
    args = ap.parse_args()
    show = lambda e, n, tl, vl, bl: print(f"  epoch {e:2d}/{n}  train {tl:.4f}  held-out {vl:.4f}  best {bl:.4f}")
    r = finetune(args.device, args.base, args.epochs, progress=show)
    print(f"\n{len(r['sessions'])} sessions, {r['windows']} windows")
    print(f"base : loss {r['base_loss']:.4f}  {format_metrics(r['base_metrics'])}")
    print(f"tuned: loss {r['tuned_loss']:.4f}  {format_metrics(r['tuned_metrics'])}")
    if args.out:
        gru_np.save(args.out, r["tuned"], device=args.device)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
