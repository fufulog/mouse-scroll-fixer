#!/usr/bin/env python3
"""Live wheel-jitter filter.

Grabs the real mouse exclusively (EVIOCGRAB), creates a virtual copy through
/dev/uinput and forwards every event. Wheel reports go through the streaming
GRU plus the hold/release policy; everything else passes untouched.
If this process exits for any reason the kernel drops the grab and the real
mouse works directly again.

  uv run python live.py              # filter live
  uv run python live.py --dry-run    # no grab: just print what it would do
"""
import argparse, array, fcntl, json, math, os, struct, sys, time

import numpy as np
import torch

from dataset import GAP_CAP_MS

DEVICE = "/dev/input/by-id/usb-YICHIP_2.4G_Receiver_b120300001-if01-event-mouse"
HERE = os.path.dirname(os.path.abspath(__file__))
EVENT = struct.Struct("llHHi")
EV_SYN, EV_KEY, EV_REL, EV_MSC = 0x00, 0x01, 0x02, 0x04
REL_WHEEL, REL_WHEEL_HI_RES = 0x08, 0x0B
EV_MAX, KEY_MAX, REL_MAX, MSC_MAX = 0x1F, 0x2FF, 0x0F, 0x07
EVIOCGRAB = 0x40044590
EVIOCSCLOCKID = 0x400445A0
UI_SET_BIT = {EV_KEY: 0x40045565, EV_REL: 0x40045566, EV_MSC: 0x40045568}
UI_SET_EVBIT, UI_DEV_SETUP, UI_DEV_CREATE, UI_DEV_DESTROY = 0x40045564, 0x405C5503, 0x5501, 0x5502


def eviocgbit(ev, nbits):
    nbytes = nbits // 8 + 1
    return 0x80000000 | (nbytes << 16) | 0x4500 | (0x20 + ev), nbytes


def supported(fd, ev, maxcode):
    req, nbytes = eviocgbit(ev, maxcode)
    buf = array.array("B", bytes(nbytes))
    fcntl.ioctl(fd, req, buf)
    return [c for c in range(maxcode + 1) if buf[c // 8] >> (c % 8) & 1]


def make_virtual(src_fd):
    """Create a uinput mouse with the same key/rel/msc capabilities as the source."""
    ufd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    for ev, maxcode in ((EV_KEY, KEY_MAX), (EV_REL, REL_MAX), (EV_MSC, MSC_MAX)):
        codes = supported(src_fd, ev, maxcode)
        if codes:
            fcntl.ioctl(ufd, UI_SET_EVBIT, ev)
            for c in codes:
                fcntl.ioctl(ufd, UI_SET_BIT[ev], c)
    setup = struct.pack("HHHH80sI", 0x03, 0x3515, 0x3020, 1, b"MouseScrollFixer virtual mouse", 0)
    fcntl.ioctl(ufd, UI_DEV_SETUP, setup)
    fcntl.ioctl(ufd, UI_DEV_CREATE)
    return ufd


class StreamingGRU:
    """NumPy single-step GRU, numerically identical to the trained torch module."""

    def __init__(self, path):
        sd = torch.load(path)["state"]
        g = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
        self.w_ih, self.w_hh = g["gru.weight_ih_l0"], g["gru.weight_hh_l0"]
        self.b_ih, self.b_hh = g["gru.bias_ih_l0"], g["gru.bias_hh_l0"]
        self.w_out, self.b_out = g["head.weight"][0], g["head.bias"][0]
        self.h = np.zeros(self.w_hh.shape[1])
        self.prev_t = None

    def step(self, t_ns, v):
        gap = GAP_CAP_MS if self.prev_t is None else min((t_ns - self.prev_t) / 1e6, GAP_CAP_MS)
        self.prev_t = t_ns
        x = np.array([v / 4.0, math.log1p(gap) / math.log1p(GAP_CAP_MS), 1.0 if v > 0 else -1.0])
        H = len(self.h)
        gi, gh = self.w_ih @ x + self.b_ih, self.w_hh @ self.h + self.b_hh
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        z = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - z) * n + z * self.h
        return 1 / (1 + math.exp(-(self.w_out @ self.h + self.b_out)))


class HoldRelease:
    """Live version of policy.hold_release: returns the wheel notches to emit now."""

    def __init__(self, hold_ms):
        self.hold_ns = hold_ms * 1_000_000
        self.held = []  # (t_ns, v)

    def feed(self, t_ns, v, suppress):
        self.held = [(t, hv) for t, hv in self.held if t_ns - self.held[0][0] <= self.hold_ns]
        if suppress:
            self.held.append((t_ns, v))
            return []
        released = [hv for _, hv in self.held if hv * v > 0]
        self.held = [(t, hv) for t, hv in self.held if hv * v < 0]
        return released + [v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "gru_h16.pt"))
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--hold-ms", type=float, default=80)
    ap.add_argument("--dry-run", action="store_true", help="don't grab; print decisions only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model, policy = StreamingGRU(args.model), HoldRelease(args.hold_ms)
    fd = os.open(DEVICE, os.O_RDONLY)
    fcntl.ioctl(fd, EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
    ufd = None
    if not args.dry_run:
        ufd = make_virtual(fd)
        time.sleep(0.3)  # let the compositor pick up the virtual device before we grab
        fcntl.ioctl(fd, EVIOCGRAB, 1)

    os.makedirs(os.path.join(HERE, "live_logs"), exist_ok=True)
    log_path = os.path.join(HERE, "live_logs", time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    log = open(log_path, "w")
    log.write(json.dumps({"thr": args.thr, "hold_ms": args.hold_ms, "model": args.model,
                          "dry_run": args.dry_run}) + "\n")
    print(f"{'DRY RUN' if args.dry_run else 'FILTERING'}  thr={args.thr} hold={args.hold_ms}ms  "
          f"log={log_path}\nCtrl+C to stop (the real mouse comes back immediately).", flush=True)

    frame, passed, suppressed = [], 0, 0
    try:
        while True:
            data = os.read(fd, EVENT.size * 64)
            for off in range(0, len(data), EVENT.size):
                sec, usec, typ, code, value = EVENT.unpack_from(data, off)
                if typ != EV_SYN:
                    frame.append((typ, code, value))
                    continue
                t_ns = sec * 1_000_000_000 + usec * 1000
                wheel = next((v for ty, c, v in frame if ty == EV_REL and c == REL_WHEEL), None)
                out = [e for e in frame if not (e[0] == EV_REL and e[1] in (REL_WHEEL, REL_WHEEL_HI_RES))]
                if wheel is not None:
                    p = model.step(t_ns, wheel)
                    emit = policy.feed(t_ns, wheel, p > args.thr)
                    total = sum(emit)
                    if total:
                        out += [(EV_REL, REL_WHEEL, total), (EV_REL, REL_WHEEL_HI_RES, total * 120)]
                    passed += bool(emit)
                    suppressed += not emit
                    log.write(json.dumps([t_ns, wheel, round(p, 4), emit]) + "\n")
                    if not args.quiet:
                        tag = "SUPPRESS" if not emit else ("release+" if len(emit) > 1 else "pass")
                        print(f"  {wheel:+d}  p={p:.2f}  {tag:<8} emit={emit}   (passed {passed} / suppressed {suppressed})", flush=True)
                if ufd is not None:
                    buf = b"".join(EVENT.pack(0, 0, ty, c, v) for ty, c, v in out)
                    os.write(ufd, buf + EVENT.pack(0, 0, EV_SYN, 0, 0))
                frame = []
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        if ufd is not None:
            fcntl.ioctl(fd, EVIOCGRAB, 0)
            fcntl.ioctl(ufd, UI_DEV_DESTROY)
        print(f"\nstopped. passed {passed}, suppressed {suppressed}. log: {log_path}")


if __name__ == "__main__":
    sys.exit(main())
