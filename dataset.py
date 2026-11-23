"""Turn recorded sessions into per-report training examples.

Each REL_WHEEL report gets a label: 1 = jitter (suppress), 0 = intended (pass).
  down/up sessions: jitter = sign opposite to the session's intent.
  sequence sessions: intent switches at the reaction onset, the first run of
    ONSET_RUN consecutive reports in the cued direction after a cue (cues lead
    the real reaction by ~400 ms, so the cue time itself is not the switch).
    After a STOP cue the previous direction stays the intent.
Rest sessions have no wheel reports and are skipped.
"""
import glob, json, math, os

SESSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
EV_REL, REL_WHEEL = 0x02, 0x08
ONSET_RUN = 3
INTENT_SIGN = {"up": 1, "down": -1}
GAP_CAP_MS = 2000.0


def load(path):
    with open(path) as f:
        meta = json.loads(f.readline())
        events = [json.loads(line) for line in f]
    ticks = [(t, v) for t, typ, code, v in events if typ == EV_REL and code == REL_WHEEL]
    return meta, ticks


def sequence_intents(meta, ticks):
    """Intent sign per report for a cued sequence, switching at reaction onset."""
    switches = []  # (t_ns, sign)
    for cue in meta["cues"]:
        want = INTENT_SIGN.get(cue["intent"])
        if not want:
            continue
        later = [(t, v) for t, v in ticks if t >= cue["t_ns"]]
        for i in range(len(later) - ONSET_RUN + 1):
            if all(v * want > 0 for _, v in later[i:i + ONSET_RUN]):
                switches.append((later[i][0], want))
                break
    intents, sign, k = [], 0, 0
    for t, _ in ticks:
        while k < len(switches) and switches[k][0] <= t:
            sign = switches[k][1]
            k += 1
        intents.append(sign)
    return intents


def features(ticks):
    """Per report: [signed notches / 4, log gap since previous report, sign]."""
    feats, prev_t = [], None
    for t, v in ticks:
        gap = GAP_CAP_MS if prev_t is None else min((t - prev_t) / 1e6, GAP_CAP_MS)
        feats.append([v / 4.0, math.log1p(gap) / math.log1p(GAP_CAP_MS), 1.0 if v > 0 else -1.0])
        prev_t = t
    return feats


def session_examples(path):
    """Returns (features, labels, notches, name) or None for sessions without intent."""
    meta, ticks = load(path)
    mode = meta["mode"]
    if not ticks:
        return None
    if mode in INTENT_SIGN:
        intents = [INTENT_SIGN[mode]] * len(ticks)
    elif mode == "sequence":
        intents = sequence_intents(meta, ticks)
    else:
        return None
    keep = [i for i, s in enumerate(intents) if s != 0]  # drop reports before the first onset
    labels = [1 if ticks[i][1] * intents[i] < 0 else 0 for i in range(len(ticks))]
    feats = features(ticks)
    first = keep[0] if keep else len(ticks)
    return {"name": os.path.basename(path), "mode": mode, "device": meta.get("device"), "feats": feats, "labels": labels,
            "notches": [abs(v) for _, v in ticks], "first_labelled": first}


def all_sessions():
    paths = sorted(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    return [s for s in (session_examples(p) for p in paths) if s]
