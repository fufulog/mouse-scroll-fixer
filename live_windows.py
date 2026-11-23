#!/usr/bin/env python3
"""Live wheel-jitter filter for Windows. Stock Python only (no NumPy, no driver).

How it works: a low-level mouse hook (WH_MOUSE_LL) sees every wheel event
before any window does. Each notch-sized event runs through the GRU and the
hold/release policy:
  pass            -> the original event continues untouched
  suppress        -> the event is swallowed (and held for up to hold_ms)
  pass + release  -> swallowed, then re-sent with SendInput as one combined event
Our own SendInput events are recognised by a marker and passed straight through.
Events that are not whole notches (precision touchpads, true hi-res wheels)
pass untouched.

  py live_windows.py                         # most recently trained profile in windows/profiles
  py live_windows.py --weights windows\\profiles\\X.json --dry-run
Ctrl+C stops it; Windows removes the hook when the process exits.
"""
import argparse, ctypes, glob, json, math, os, sys, time
from ctypes import wintypes

if sys.platform != "win32":
    raise SystemExit("live_windows.py is for Windows; on Linux use live.py")

from dataset import FeatureExtractor
from policy import HoldRelease

HERE = os.path.dirname(os.path.abspath(__file__))
WH_MOUSE_LL, HC_ACTION, WM_MOUSEWHEEL = 14, 0, 0x020A
LLMHF_INJECTED, WHEEL_DELTA = 0x01, 120
INPUT_MOUSE, MOUSEEVENTF_WHEEL = 0, 0x0800
PM_REMOVE, QS_ALLINPUT, WAIT_MS = 0x0001, 0x04FF, 100
MARKER = 0x4D534621  # "MSF!" in dwExtraInfo tags our own injected events

ULONG_PTR = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class INPUT(ctypes.Structure):
    # MOUSEINPUT is the largest member of the INPUT union on both 32- and 64-bit.
    _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


def win32():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.CallNextHookEx.restype = LRESULT
    user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.MsgWaitForMultipleObjects.argtypes = (wintypes.DWORD, ctypes.c_void_p, wintypes.BOOL,
                                                 wintypes.DWORD, wintypes.DWORD)
    user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                    wintypes.UINT, wintypes.UINT)
    kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    return user32, kernel32


class PureGRU:
    """Single-step GRU in plain Python (16 hidden units: ~20 µs per report)."""

    def __init__(self, doc):
        self.w_ih, self.w_hh = doc["w_ih"], doc["w_hh"]
        self.b_ih, self.b_hh = doc["b_ih"], doc["b_hh"]
        self.w_out, self.b_out = doc["w_out"], doc["b_out"]
        self.H = len(self.w_hh[0])
        self.h = [0.0] * self.H
        self.fe = FeatureExtractor()

    def step(self, t_ns, v):
        x, h, H = self.fe.step(t_ns, v), self.h, self.H
        gi = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.w_ih, self.b_ih)]
        gh = [sum(w * hj for w, hj in zip(row, h)) + b for row, b in zip(self.w_hh, self.b_hh)]
        sig = lambda a: 1.0 / (1.0 + math.exp(-a))
        r = [sig(gi[k] + gh[k]) for k in range(H)]
        z = [sig(gi[H + k] + gh[H + k]) for k in range(H)]
        n = [math.tanh(gi[2 * H + k] + r[k] * gh[2 * H + k]) for k in range(H)]
        self.h = [(1 - z[k]) * n[k] + z[k] * h[k] for k in range(H)]
        return sig(sum(w * hk for w, hk in zip(self.w_out, self.h)) + self.b_out)


def default_weights():
    def trained_at(path):
        with open(path) as f:
            return json.load(f).get("trained_at", 0)

    found = sorted(glob.glob(os.path.join(HERE, "windows", "profiles", "*.json")), key=trained_at)
    if not found:
        raise SystemExit("No profile in windows/profiles. Export one on Linux: "
                         "uv run python export_windows.py profiles/<device>.npz")
    return found[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="default: most recently trained windows/profiles/*.json")
    ap.add_argument("--thr", type=float, help="default: the profile's value")
    ap.add_argument("--hold-ms", type=float, help="default: the profile's value")
    ap.add_argument("--dry-run", action="store_true", help="decide and print, but never block or inject")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = args.weights or default_weights()
    with open(path) as f:
        doc = json.load(f)
    thr = args.thr if args.thr is not None else doc.get("thr", 0.7)
    hold_ms = args.hold_ms if args.hold_ms is not None else doc.get("hold_ms", 100)
    model, policy = PureGRU(doc), HoldRelease(hold_ms)
    user32, kernel32 = win32()

    os.makedirs(os.path.join(HERE, "live_logs"), exist_ok=True)
    log_path = os.path.join(HERE, "live_logs", "win-" + time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    log = open(log_path, "w")
    log.write(json.dumps({"platform": "windows", "weights": path, "thr": thr, "hold_ms": hold_ms,
                          "dry_run": args.dry_run}) + "\n")
    stats = {"passed": 0, "suppressed": 0, "released": 0, "non_notch": 0}

    def inject(total):
        inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, (total * WHEEL_DELTA) & 0xFFFFFFFF,
                                                    MOUSEEVENTF_WHEEL, 0, MARKER))
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    @HOOKPROC
    def hook(n_code, w_param, l_param):
        if n_code != HC_ACTION or w_param != WM_MOUSEWHEEL:
            return user32.CallNextHookEx(None, n_code, w_param, l_param)
        info = ctypes.cast(l_param, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
        if info.dwExtraInfo == MARKER or info.flags & LLMHF_INJECTED:
            return user32.CallNextHookEx(None, n_code, w_param, l_param)
        delta = ctypes.c_short(info.mouseData >> 16).value
        if delta == 0 or delta % WHEEL_DELTA:
            stats["non_notch"] += 1
            return user32.CallNextHookEx(None, n_code, w_param, l_param)
        t_ns = time.perf_counter_ns()
        v = delta // WHEEL_DELTA
        p = model.step(t_ns, v)
        emit = policy.feed(t_ns, v, p > thr)
        log.write(json.dumps([t_ns, v, round(p, 4), emit]) + "\n")
        if not emit:
            stats["suppressed"] += 1
            tag = "SUPPRESS"
        elif emit == [v]:
            stats["passed"] += 1
            tag = "pass"
        else:
            stats["released"] += 1
            tag = "release+"
        if not args.quiet:
            print(f"  {v:+d}  p={p:.2f}  {tag:<8} emit={emit}", flush=True)
        if args.dry_run or emit == [v]:
            return user32.CallNextHookEx(None, n_code, w_param, l_param)
        if emit:
            inject(sum(emit))
        return 1  # swallow the original

    hhook = user32.SetWindowsHookExW(WH_MOUSE_LL, hook, kernel32.GetModuleHandleW(None), 0)
    if not hhook:
        raise SystemExit(f"SetWindowsHookExW failed: error {ctypes.get_last_error()}")
    print(f"{'DRY RUN' if args.dry_run else 'FILTERING'}  thr={thr} hold={hold_ms}ms\n"
          f"weights={os.path.relpath(path, HERE)}\nlog={log_path}\nCtrl+C to stop.", flush=True)
    msg = wintypes.MSG()
    try:
        while True:  # short waits so Ctrl+C is noticed; hooks run while we pump messages
            user32.MsgWaitForMultipleObjects(0, None, False, WAIT_MS, QS_ALLINPUT)
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        pass
    finally:
        user32.UnhookWindowsHookEx(hhook)
        log.close()
        print(f"\nstopped. {stats}. log: {log_path}")


if __name__ == "__main__":
    main()
