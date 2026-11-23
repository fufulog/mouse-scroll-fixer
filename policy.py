"""Output policies: turn per-report suppress decisions into what reaches the OS."""


def drop(ticks, suppress):
    """Suppressed reports are deleted. Returns [(emit_t, v, src_index)]."""
    return [(t, v, i) for i, ((t, v), s) in enumerate(zip(ticks, suppress)) if not s]


def hold_release(ticks, suppress, hold_ms):
    """Suppressed reports are held. If the model later passes a report of the same
    sign within hold_ms of the oldest held one, the held reports are released with
    it (a real reversal arrives late, not shortened); otherwise they expire."""
    out, held = [], []
    for i, ((t, v), s) in enumerate(zip(ticks, suppress)):
        held = [j for j in held if t - ticks[j][0] <= hold_ms * 1e6]
        if s:
            held.append(i)
            continue
        same = [j for j in held if ticks[j][1] * v > 0]
        out += [(t, ticks[j][1], j) for j in same]
        held = [j for j in held if ticks[j][1] * v < 0]
        out.append((t, v, i))
    return out


class HoldRelease:
    """Live version of policy.hold_release: returns the wheel notches to emit now."""

    def __init__(self, hold_ms):
        self.hold_ns = hold_ms * 1_000_000
        self.held = []  # (t_ns, v)

    def feed(self, t_ns, v, suppress):
        self.held = [(t, hv) for t, hv in self.held if t_ns - t <= self.hold_ns]
        if suppress:
            self.held.append((t_ns, v))
            return []
        released = [hv for _, hv in self.held if hv * v > 0]
        self.held = [(t, hv) for t, hv in self.held if hv * v < 0]
        return released + [v]
