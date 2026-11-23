"""Turn recorded sessions into per-report training examples.

Each REL_WHEEL report gets a label: 1 = jitter (suppress), 0 = intended (pass).
  down/up sessions: jitter = sign opposite to the session's intent.
  sequence sessions: intent switches at the reaction onset, the first run of
    ONSET_RUN consecutive reports in the cued direction after a cue (cues lead
    the real reaction by ~400 ms, so the cue time itself is not the switch).
    After a STOP cue the previous direction stays the intent.
Rest sessions have no wheel reports and are skipped.
"""
import glob, json, math, os, random

SESSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
EV_REL, REL_WHEEL = 0x02, 0x08
ONSET_RUN = 3
INTENT_SIGN = {"up": 1, "down": -1}
GAP_CAP_MS = 2000.0
BURST_GAP_MS = 100.0
LOG_GAP_CAP = math.log1p(GAP_CAP_MS)
LOG_MAG_CAP = math.log1p(4.0)
LOG_RUN_CAP = math.log1p(8.0)
IN_DIM = 5


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


class FeatureExtractor:
    """Causal per-report features: [sign, log_mag, log_gap, same_sign, log_burst_run]."""

    def __init__(self):
        self.prev_t = None
        self.prev_s = 0
        self.burst_run = 0

    def step(self, t_ns, v):
        gap = GAP_CAP_MS if self.prev_t is None else min(max((t_ns - self.prev_t) / 1e6, 0.0), GAP_CAP_MS)
        si = 1 if v > 0 else -1
        same = 1.0 if si == self.prev_s else -1.0
        self.burst_run = self.burst_run + 1 if (si == self.prev_s and gap < BURST_GAP_MS) else 1
        self.prev_t = t_ns
        self.prev_s = si
        return [
            float(si),
            math.log1p(abs(v)) / LOG_MAG_CAP,
            math.log1p(gap) / LOG_GAP_CAP,
            same,
            math.log1p(min(self.burst_run, 8)) / LOG_RUN_CAP,
        ]


def features(ticks):
    """Per report: [sign, log_mag, log_gap, same_sign, log_burst_run]."""
    fe = FeatureExtractor()
    return [fe.step(t, v) for t, v in ticks]


def session_examples(path):
    """Returns session dict or None for sessions without intent."""
    meta, ticks = load(path)
    mode = meta["mode"]
    if not ticks:
        return None
    if mode in INTENT_SIGN:
        intents = [INTENT_SIGN[mode]] * len(ticks)
    elif mode in ("sequence", "model_test"):
        intents = sequence_intents(meta, ticks)
    else:
        return None
    keep = [i for i, s in enumerate(intents) if s != 0]  # drop reports before the first onset
    labels = [1 if ticks[i][1] * intents[i] < 0 else 0 for i in range(len(ticks))]
    feats = features(ticks)
    first = keep[0] if keep else len(ticks)
    return {"name": os.path.basename(path), "mode": mode, "device": meta.get("device"), "feats": feats, "labels": labels,
            "notches": [abs(v) for _, v in ticks], "first_labelled": first, "ticks": ticks, "intents": intents,
            "against_model": meta.get("against_model"),
            "include_in_train": meta.get("include_in_train", mode != "model_test")}


def synthetic_reversals(sessions, count, window=64, burn_in=16, seed=0, end_indices=None):
    """Splice single-intent training spans into synthetic reversal windows."""
    if count <= 0:
        return [], [], []
    spans = {1: [], -1: []}
    for s in sessions:
        ticks, ints, labs = s["ticks"], s["intents"], s["labels"]
        limit = end_indices[s["name"]] if end_indices else len(ticks)
        i = s["first_labelled"]
        while i < limit:
            sgn = ints[i]
            j = i
            while j < limit and ints[j] == sgn:
                j += 1
            if sgn in spans and j - i >= 20:
                spans[sgn].append((ticks[i:j], labs[i:j]))
            i = j
    if not spans[1] or not spans[-1]:
        return [], [], []

    starts = {1: [], -1: []}
    for sgn in (1, -1):
        for sp_i, (tk, lb) in enumerate(spans[sgn]):
            for idx in range(len(tk)):
                if lb[idx] == 0 and (
                    idx == 0
                    or (tk[idx][0] - tk[idx - 1][0]) > 100e6
                    or (idx + 2 < len(lb) and lb[idx + 1] == 0 and lb[idx + 2] == 0)
                ):
                    starts[sgn].append((sp_i, idx))

    rng = random.Random(seed)
    xs, ys, ws = [], [], []
    for _ in range(count):
        s1 = rng.choice((1, -1))
        s2 = -s1
        k = rng.randint(burn_in + 6, window - 14)
        pre_cands = [(tk, lb) for tk, lb in spans[s1] if len(tk) >= k]
        if not pre_cands:
            continue
        sp1_tk, sp1_lb = rng.choice(pre_cands)
        p0 = rng.randint(0, len(sp1_tk) - k)
        pre_tk, pre_lb = sp1_tk[p0:p0 + k], sp1_lb[p0:p0 + k]
        need = window - k
        cand = [(si, idx) for si, idx in starts[s2] if len(spans[s2][si][0]) - idx >= need]
        if not cand:
            continue
        si, idx = rng.choice(cand)
        sp2_tk, sp2_lb = spans[s2][si]
        suf_tk, suf_lb = sp2_tk[idx:idx + need], sp2_lb[idx:idx + need]
        gap_ms = (
            math.exp(rng.uniform(math.log(6.0), math.log(80.0)))
            if rng.random() < 0.55
            else math.exp(rng.uniform(math.log(80.0), math.log(480.0)))
        )
        shift = pre_tk[-1][0] + int(gap_ms * 1e6) - suf_tk[0][0]
        comb_tk = list(pre_tk) + [(t + shift, v) for t, v in suf_tk]
        comb_lb = list(pre_lb) + list(suf_lb)
        xs.append(features(comb_tk))
        ys.append(comb_lb)
        ws.append([0.0 if i < burn_in else float(abs(comb_tk[i][1])) for i in range(window)])
    return xs, ys, ws


def all_sessions(include_model_tests=False):
    paths = sorted(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    out = [s for s in (session_examples(p) for p in paths) if s]
    return out if include_model_tests else [s for s in out if s["mode"] != "model_test"]
