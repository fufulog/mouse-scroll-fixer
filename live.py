#!/usr/bin/env python3
"""Live wheel-jitter filter.

Grabs the real mouse exclusively (EVIOCGRAB), creates a virtual copy through
/dev/uinput and forwards every event. Wheel reports go through the streaming
GRU plus the hold/release policy; everything else passes untouched.
If this process exits for any reason the kernel drops the grab and the real
mouse works directly again.

  python live.py                     # filter live (device's profile, else base weights)
  python live.py --device PATH --weights FILE
  python live.py --dry-run           # no grab: just print what it would do
"""
import argparse, array, fcntl, json, os, struct, sys, time

import dataset, devices, gru_np
from policy import HoldRelease

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


def source_ids(path):
    sysdir = f"/sys/class/input/{os.path.basename(os.path.realpath(path))}/device/id"
    try:
        return tuple(int(open(f"{sysdir}/{k}").read(), 16) for k in ("bustype", "vendor", "product"))
    except OSError:
        return (0x03, 0x1, 0x1)


def make_virtual(src_fd, src_path):
    """Create a uinput mouse with the same key/rel/msc capabilities as the source."""
    ufd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    for ev, maxcode in ((EV_KEY, KEY_MAX), (EV_REL, REL_MAX), (EV_MSC, MSC_MAX)):
        codes = supported(src_fd, ev, maxcode)
        if codes:
            fcntl.ioctl(ufd, UI_SET_EVBIT, ev)
            for c in codes:
                fcntl.ioctl(ufd, UI_SET_BIT[ev], c)
    bus, vendor, product = source_ids(src_path)
    setup = struct.pack("HHHH80sI", bus, vendor, product, 1, b"MouseScrollFixer virtual mouse", 0)
    fcntl.ioctl(ufd, UI_DEV_SETUP, setup)
    fcntl.ioctl(ufd, UI_DEV_CREATE)
    return ufd


def pick_device(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"Device not found: {explicit}")
        if not os.access(explicit, os.R_OK):
            raise SystemExit(f"No read access to {explicit}.\nGrant it: {devices.access_hint(explicit)}")
        return explicit
    mice = devices.wheel_mice()
    if not mice:
        raise SystemExit("No mouse with a scroll wheel found.")
    if len(mice) == 1:
        m = mice[0]
        if not m["readable"]:
            raise SystemExit(f"No read access to {m['name']} ({m['path']}).\nGrant it: {devices.access_hint(m['path'])}")
        return m["path"]
    readable = [m for m in mice if m["readable"]]
    if len(readable) == 1:
        return readable[0]["path"]
    lines = ["Multiple wheel mice found — pass --device PATH or run app.py:"]
    for m in mice:
        hint = "readable" if m["readable"] else f"no access: {devices.access_hint(m['path'])}"
        lines.append(f"  {m['name']}: {m['path']} ({hint})")
    raise SystemExit("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="default: auto-detect connected wheel mouse")
    ap.add_argument("--weights", help="default: the device's profile, else weights/base.npz")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--hold-ms", type=float, default=100)
    ap.add_argument("--dry-run", action="store_true", help="don't grab; print decisions only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    if not args.dry_run and not os.access("/dev/uinput", os.W_OK):
        raise SystemExit("No write access to /dev/uinput.\nGrant it: sudo setfacl -m u:$USER:rw /dev/uinput")

    weights = args.weights or devices.weights_for(device)
    params = gru_np.load(weights)
    in_dim = params["w_ih"].shape[1]
    if in_dim != dataset.IN_DIM:
        raise SystemExit(f"incompatible weights {weights}: expects {in_dim} features, need {dataset.IN_DIM} "
                         f"(re-run train.py --all --export-base or re-tune)")
    model, policy = gru_np.Streaming(params), HoldRelease(args.hold_ms)
    fd = os.open(device, os.O_RDONLY)
    fcntl.ioctl(fd, EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
    ufd = None
    if not args.dry_run:
        ufd = make_virtual(fd, device)
        time.sleep(0.3)  # let the compositor pick up the virtual device before we grab
        fcntl.ioctl(fd, EVIOCGRAB, 1)

    os.makedirs(os.path.join(HERE, "live_logs"), exist_ok=True)
    log_path = os.path.join(HERE, "live_logs", time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    log = open(log_path, "w", buffering=1)
    log.write(json.dumps({"thr": args.thr, "hold_ms": args.hold_ms, "weights": weights, "device": device,
                          "dry_run": args.dry_run}) + "\n")
    print(f"{'DRY RUN' if args.dry_run else 'FILTERING'}  thr={args.thr} hold={args.hold_ms}ms  device={device}\n"
          f"weights={os.path.relpath(weights, HERE)}\nlog={log_path}\nCtrl+C to stop (the real mouse comes back immediately).", flush=True)

    frame, passed, suppressed, warned_hires = [], 0, 0, False
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
                hires = next((v for ty, c, v in frame if ty == EV_REL and c == REL_WHEEL_HI_RES), None)
                if hires is not None and not warned_hires and (wheel is None or hires != wheel * 120):
                    warned_hires = True
                    print("  note: this mouse has true hi-res scrolling; the filter only sees whole notches "
                          "and re-emits them as such, so scrolling will be coarser.", flush=True)
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
