#!/usr/bin/env python3
"""Summarise recorded sessions: jitter rate, timing of reversals, reaction time."""
import glob, json, os, statistics, sys

EV_REL, REL_WHEEL, REL_WHEEL_HI_RES = 0x02, 0x08, 0x0B
PAUSE_NS = 150_000_000


def load(path):
    with open(path) as f:
        meta = json.loads(f.readline())
        events = [json.loads(line) for line in f]
    return meta, events


def wheel(events, code):
    return [(t, v) for t, typ, c, v in events if typ == EV_REL and c == code]


def intent_at(cues, t):
    current = None
    for cue in cues:
        if cue["t_ns"] > t:
            break
        current = cue["intent"]
    return current


def bursts(hi):
    """Split ticks into bursts separated by pauses (no tick for > PAUSE_NS)."""
    out, cur = [], []
    for t, v in hi:
        if cur and t - cur[-1][0] > PAUSE_NS:
            out.append(cur)
            cur = []
        cur.append((t, v))
    return out + [cur] if cur else out


def burst_report(hi, sign):
    """End-of-scroll reversal: wrong-direction ticks trailing a burst, right before a pause.
    A lone wrong tick that forms its own burst counts against the burst before it."""
    runs = bursts(hi)
    ended_reversed, delays, starts_wrong = 0, [], 0
    for k, run in enumerate(runs):
        good = [t for t, v in run if v * sign > 0]
        if run[0][1] * sign < 0 and good:
            starts_wrong += 1
        tail = []
        for t, v in reversed(run):
            if v * sign > 0:
                break
            tail.append(t)
        if not good:  # whole burst is wrong ticks: attach to previous burst's end
            prev_good = [t for t, v in runs[k - 1] if v * sign > 0] if k else []
            if prev_good:
                ended_reversed += 1
                delays += [(t - prev_good[-1]) / 1e6 for t in tail]
            continue
        if tail:
            ended_reversed += 1
            delays += [(t - good[-1]) / 1e6 for t in tail]
    real = sum(1 for r in runs if any(v * sign > 0 for _, v in r))
    print(f"   bursts (pause > {PAUSE_NS // 1_000_000}ms): {real}   ending with reversal: {ended_reversed}"
          f"   starting with a wrong tick: {starts_wrong}")
    if delays:
        print(f"   reversal delay after last good tick ms: min {min(delays):.0f}  median {statistics.median(delays):.0f}"
              f"  max {max(delays):.0f}   [{' '.join(f'{d:.0f}' for d in delays[:30])}]")


def summarise(path):
    meta, events = load(path)
    hi = wheel(events, REL_WHEEL_HI_RES)
    lo = wheel(events, REL_WHEEL)
    print(f"\n== {os.path.basename(path)}  ({meta['mode']}, {len(events)} raw events)")
    if not hi:
        print("   no hi-res wheel events")
        return
    span = (hi[-1][0] - hi[0][0]) / 1e9
    values = sorted({v for _, v in hi})
    print(f"   hi-res: {len(hi)} events over {span:.2f}s, distinct values {values}")
    print(f"   notch REL_WHEEL: {len(lo)} events, values {sorted({v for _, v in lo})}")
    gaps = [(b[0] - a[0]) / 1e6 for a, b in zip(hi, hi[1:])]
    if gaps:
        print(f"   inter-event ms: min {min(gaps):.1f}  median {statistics.median(gaps):.1f}  max {max(gaps):.1f}")

    intended = {"down": -1, "up": 1}
    if meta["mode"] in intended:
        sign = intended[meta["mode"]]
        wrong = [(i, t, v) for i, (t, v) in enumerate(hi) if v * sign < 0]
        print(f"   against intent: {len(wrong)}/{len(hi)} ({100 * len(wrong) / len(hi):.1f}%)")
        burst_report(hi, sign)
        for i, t, v in wrong[:15]:
            prev = (t - hi[i - 1][0]) / 1e6 if i else float("nan")
            nxt = (hi[i + 1][0] - t) / 1e6 if i + 1 < len(hi) else float("nan")
            ctx = " ".join(f"{hv:+d}" for _, hv in hi[max(0, i - 4):i + 5])
            print(f"     t={(t - hi[0][0]) / 1e6:8.1f}ms  v={v:+d}  gap_prev={prev:5.1f}ms gap_next={nxt:5.1f}ms  ctx[{ctx}]")
    elif meta["mode"] == "rest":
        print(f"   spurious events while resting: {len(hi)}")
    elif meta["cues"]:
        cues = meta["cues"]
        reversal_delays = []
        prev_want = None
        for cue, nxt in zip(cues, cues[1:] + [None]):
            want = intended.get(cue["intent"])
            end = nxt["t_ns"] if nxt else float("inf")
            seg = [(t, v) for t, v in hi if cue["t_ns"] <= t < end]
            if want:
                first = next(((t - cue["t_ns"]) / 1e6 for t, v in seg if v * want > 0), None)
                wrong = sum(1 for _, v in seg if v * want < 0)
                rt = f"{first:6.0f}ms" if first is not None else "   n/a"
                print(f"   cue {cue['intent']:>4}: reaction {rt}  events {len(seg):4d}  opposite-sign {wrong}")
                prev_want = want
                continue
            # STOP: same-direction ticks = reaction lag (fine); opposite ticks = end-of-scroll reversal.
            lag = [t for t, v in seg if prev_want and v * prev_want > 0]
            last_good = max([t for t, v in hi if t < cue["t_ns"] and prev_want and v * prev_want > 0] + lag, default=None)
            rev = [(t, v) for t, v in seg if prev_want and v * prev_want < 0]
            delays = [(t - last_good) / 1e6 for t, _ in rev] if last_good else []
            reversal_delays += delays
            lag_ms = f"{(max(lag) - cue['t_ns']) / 1e6:5.0f}ms" if lag else "    -"
            dtxt = " ".join(f"{d:.0f}" for d in delays) or "-"
            print(f"   cue stop: lag ticks {len(lag):3d} (last at {lag_ms})  REVERSALS {len(rev)}  ms after last good tick: {dtxt}")
        stops = sum(1 for c in cues if c["intent"] == "stop")
        if reversal_delays:
            print(f"   => {len(reversal_delays)} reversal ticks over {stops} stops; delay after last good tick "
                  f"min {min(reversal_delays):.0f} median {statistics.median(reversal_delays):.0f} max {max(reversal_delays):.0f} ms")
        else:
            print(f"   => no reversal ticks over {stops} stops")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "sessions", "*.jsonl")))
    if not paths:
        sys.exit("no sessions recorded yet")
    for p in paths:
        summarise(p)
