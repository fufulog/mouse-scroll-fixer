#!/usr/bin/env python3
"""Rule-based baseline: hold ticks opposite to the current direction; release them only if
they add up to >= M notches within W ms (a real reversal), otherwise drop them as jitter."""
import glob, json, os, sys

SESSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

def load(p):
    with open(p) as f:
        meta = json.loads(f.readline())
        ev = [json.loads(l) for l in f]
    return meta, [(t, v) for t, ty, c, v in ev if ty == 2 and c == 8]

def run_filter(ticks, M, W_ms):
    """Returns [(emit_t, v, src_t)]: what the filter would send to the OS."""
    out, direction, held = [], 0, []
    for t, v in ticks:
        if held and t - held[0][0] > W_ms * 1e6:
            held = []  # unconfirmed: jitter, dropped
        s = 1 if v > 0 else -1
        if direction in (0, s):
            direction = s
            held = []
            out.append((t, v, t))
            continue
        held.append((t, v))
        if sum(abs(hv) for _, hv in held) >= M:
            direction = s
            out += [(t, hv, ht) for ht, hv in held]
            held = []
    return out

def score_free(path, sign, M, W):
    _, ticks = load(path)
    out = run_filter(ticks, M, W)
    wrong_in = sum(abs(v) for _, v in ticks if v * sign < 0)
    wrong_out = sum(abs(v) for _, v, _ in out if v * sign < 0)
    good_in = sum(abs(v) for _, v in ticks if v * sign > 0)
    good_out = sum(abs(v) for _, v, _ in out if v * sign > 0)
    return wrong_in, wrong_out, good_in, good_out

def score_sequence(path, M, W):
    """Latency added at real reversals: first sustained (>=3 reports) run in the cued direction."""
    meta, ticks = load(path)
    out = run_filter(ticks, M, W)
    lat, lost = [], 0
    cues = meta["cues"]
    for cue, nxt in zip(cues, cues[1:]):
        want = {"up": 1, "down": -1}.get(cue["intent"])
        if not want:
            continue
        seg = [(t, v) for t, v in ticks if cue["t_ns"] <= t < nxt["t_ns"] + 600e6]
        for i in range(len(seg) - 2):
            if all(v * want > 0 for _, v in seg[i:i + 3]):
                onset = seg[i][0]
                emitted = [e for e, v, src in out if src == onset]
                if emitted:
                    lat.append((emitted[0] - onset) / 1e6)
                else:
                    lost += 1
                break
    return lat, lost

if __name__ == "__main__":
    down = sorted(glob.glob(os.path.join(SESSIONS, "*_down.jsonl")))[0]
    up = sorted(glob.glob(os.path.join(SESSIONS, "*_up.jsonl")))[0]
    seq = sorted(glob.glob(os.path.join(SESSIONS, "*_sequence.jsonl")))[0]
    print(" M   W  | down: jitter notches passed  good kept | up: jitter passed  good kept | seq reversals: added latency ms (max)  onsets lost")
    for M in (2, 3, 4):
        for W in (40, 80, 150):
            d = score_free(down, -1, M, W)
            u = score_free(up, 1, M, W)
            lat, lost = score_sequence(seq, M, W)
            print(f" {M}  {W:3d}  |   {d[1]:3d}/{d[0]:3d} ({100*d[1]/d[0]:4.1f}%)   {100*d[3]/d[2]:5.1f}%"
                  f"   |   {u[1]:3d}/{u[0]:3d} ({100*u[1]/u[0]:4.1f}%)  {100*u[3]/u[2]:5.1f}%"
                  f"  |   mean {sum(lat)/max(len(lat),1):5.1f}  max {max(lat, default=0):5.1f}   {lost}")
