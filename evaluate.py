#!/usr/bin/env python3
"""Score a trained model on the validation sessions under each output policy,
and list every real reversal in the sequence sessions with what the model did to it."""
import argparse, os

import numpy as np
import torch

import dataset, policy
from train import JitterGRU, split, stream_probs


def load_model(path):
    ck = torch.load(path)
    in_dim = ck.get("in_dim", ck["state"]["gru.weight_ih_l0"].shape[1])
    model = JitterGRU(ck["hidden"], in_dim)
    model.load_state_dict(ck["state"])
    return model.eval()


def evaluate(label, sessions, model, thr, emit):
    jp = jt = gl = gt = 0
    delays = []
    for s in sessions:
        _, ticks = dataset.load(os.path.join(dataset.SESSIONS, s["name"]))
        emitted = {src: et for et, _, src in emit(ticks, stream_probs(model, s) > thr)}
        for i in range(s["first_labelled"], len(ticks)):
            n = abs(ticks[i][1])
            if s["labels"][i]:
                jt += n
                jp += n if i in emitted else 0
            elif i in emitted:
                gt += n
                delays.append((emitted[i] - ticks[i][0]) / 1e6)
            else:
                gt += n
                gl += n
    late = np.array([d for d in delays if d > 0])
    lat = f"median {np.median(late):.0f} max {late.max():.0f} ms" if len(late) else "-"
    print(f"  {label:<20} jitter passed {jp:3d}/{jt} ({100 * jp / jt:4.1f}%)  good lost {gl:3d}/{gt} "
          f"({100 * gl / gt:4.2f}%)  good delayed {len(late)} reports, {lat}")


def reversals(sessions, model, thr, hold_ms=100):
    for s in sessions:
        if s["mode"] != "sequence":
            continue
        meta, ticks = dataset.load(os.path.join(dataset.SESSIONS, s["name"]))
        sup = stream_probs(model, s) > thr
        emitted = {src for _, _, src in policy.hold_release(ticks, sup, hold_ms)}
        intents = dataset.sequence_intents(meta, ticks)
        print(f"  {s['name']}")
        for i in range(1, len(ticks)):
            if intents[i] == intents[i - 1] or intents[i - 1] == 0:
                continue
            seg_end = i
            while seg_end < len(ticks) and intents[seg_end] == intents[i]:
                seg_end += 1
            j, eaten = i, 0
            while j < seg_end:
                if ticks[j][1] * intents[i] > 0:
                    if not sup[j]:
                        break
                    eaten += abs(ticks[j][1])
                j += 1
            wait = (ticks[j][0] - ticks[i][0]) / 1e6 if j < seg_end else float("nan")
            hr_lost = sum(abs(ticks[k][1]) for k in range(i, seg_end)
                          if ticks[k][1] * intents[i] > 0 and k not in emitted)
            ctx = " ".join(f"{v:+d}{'x' if sup[k] else ''}" for k, (_, v) in enumerate(ticks[i:i + 8], start=i))
            print(f"    reversal to {intents[i]:+d}: {eaten:2d} eaten / {hr_lost:2d} hr_lost, "
                  f"first pass after {wait:4.0f} ms  [{ctx}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gru_h16.pt"))
    ap.add_argument("--thr", type=float, default=0.7)
    args = ap.parse_args()
    model = load_model(args.model)
    _, val = split(dataset.all_sessions())
    print(f"validation sessions, threshold {args.thr} (x = suppressed):")
    evaluate("drop", val, model, args.thr, policy.drop)
    for h in (40, 80, 100, 150):
        evaluate(f"hold_release {h}ms", val, model, args.thr, lambda t, s, h=h: policy.hold_release(t, s, h))
    print("real reversals:")
    reversals(val, model, args.thr)


if __name__ == "__main__":
    main()
