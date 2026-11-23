#!/usr/bin/env python3
"""MouseScrollFixer: calibrate the jitter filter to a mouse and run it.

  pick mouse -> record (down, up, cued sequence) -> fine-tune -> compare -> save profile -> launch

Python 3 + NumPy only. Recordings land in sessions/, profiles in profiles/.
"""
import argparse, curses, errno, fcntl, glob, json, os, re, select, struct, sys, threading, time

import dataset, devices, finetune, gru_np, live, policy, recorder

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [("down", 1, "Scroll DOWN in bursts, stop now and then (30 s)"),
         ("up", 1, "Scroll UP in bursts, stop now and then (30 s)"),
         ("sequence", 2, "Follow the UP/DOWN arrows (~35 s each)")]
FREE_LIMIT_S = 30


def session_counts(device):
    counts = {}
    for path in glob.glob(os.path.join(recorder.OUT_DIR, "*.jsonl")):
        with open(path) as f:
            meta = json.loads(f.readline())
        if meta.get("device") == device:
            counts[meta["mode"]] = counts.get(meta["mode"], 0) + 1
    return counts


def init_colors():
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)


def put(scr, y, text, attr=0):
    h, w = scr.getmaxyx()
    if 0 <= y < h:
        scr.addstr(y, 2, text[:max(w - 4, 0)], attr)


def wait_key(scr):
    scr.nodelay(True)
    recorder.pending_keys(scr)
    scr.nodelay(False)
    return scr.getch()


def pick_device(scr):
    while True:
        mice = devices.wheel_mice()
        scr.erase()
        put(scr, 1, "MouseScrollFixer — pick a mouse", curses.A_BOLD)
        for i, m in enumerate(mice[:9]):
            state = "" if m["readable"] else "   (no access: " + devices.access_hint(m["path"]) + ")"
            put(scr, 3 + i, f"{i + 1}  {m['name']}{state}", 0 if m["readable"] else curses.A_DIM)
        if not mice:
            put(scr, 3, "No mouse with a scroll wheel found.")
        put(scr, 4 + len(mice[:9]), "r  rescan    q  quit", curses.A_DIM)
        scr.refresh()
        key = wait_key(scr)
        if key == ord("q"):
            return None
        if ord("1") <= key < ord("1") + len(mice[:9]):
            m = mice[key - ord("1")]
            if m["readable"]:
                return m


def run_finetune(scr, device):
    scr.erase()
    put(scr, 1, "Fine-tuning from base weights…", curses.A_BOLD)
    scr.refresh()

    def progress(epoch, epochs, train_loss, val, best):
        bar_w = 30
        done = int(bar_w * epoch / epochs)
        put(scr, 3, f"[{'#' * done}{'.' * (bar_w - done)}] epoch {epoch}/{epochs}")
        put(scr, 4, f"train loss {train_loss:.4f}   held-out {val:.4f}   best {best:.4f}   ")
        scr.refresh()

    r = finetune.finetune(device, finetune.BASE, progress=progress)
    better = r["tuned_loss"] < r["base_loss"]
    put(scr, 6, f"held-out = last 25% of each of {len(r['sessions'])} sessions, live hold/release policy:")
    put(scr, 8, f"base : loss {r['base_loss']:.4f}  {finetune.format_metrics(r['base_metrics'])}")
    put(scr, 9, f"tuned: loss {r['tuned_loss']:.4f}  {finetune.format_metrics(r['tuned_metrics'])}",
        curses.A_BOLD if better else 0)
    put(scr, 11, "tuned is better on held-out loss" if better else "tuned did NOT beat base; saving it is not advised",
        curses.color_pair(2 if better else 3))
    put(scr, 13, "s  save as this mouse's profile     d  discard")
    scr.refresh()
    while True:
        key = wait_key(scr)
        if key == ord("s"):
            os.makedirs(devices.PROFILES, exist_ok=True)
            gru_np.save(devices.profile_path(device), r["tuned"], device=device)
            return "profile saved"
        if key == ord("d"):
            return "discarded"


def latest_live_log(log_dir=os.path.join(HERE, "live_logs")):
    paths = glob.glob(os.path.join(log_dir, "*.jsonl"))
    if not paths:
        return None, None
    path = max(paths, key=os.path.getmtime)
    try:
        with open(path) as f:
            meta = json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return None, None
    return path, meta


class LiveLogTailer:
    """Align cued tests with a running live.py log via CLOCK_MONOTONIC timestamps."""

    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "rb") as f:
            self.meta = json.loads(f.readline().decode("utf-8"))
        self.start_offset = os.path.getsize(log_path)
        self.pos = self.start_offset
        self.decisions = []

    def poll(self):
        with open(self.log_path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read()
        nl = chunk.rfind(b"\n")
        if nl != -1:
            complete = chunk[:nl + 1]
            self.pos += len(complete)
            for line in complete.decode("utf-8").splitlines():
                if line.strip():
                    try:
                        self.decisions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return list(self.decisions)

    def collect(self, started_ns, ended_ns):
        with open(self.log_path, "rb") as f:
            f.seek(self.start_offset)
            data = f.read()
        rows = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        decisions = [r for r in rows if started_ns <= r[0] <= ended_ns]
        events = []
        for t_ns, wheel, _, _ in decisions:
            events.append([t_ns, live.EV_REL, live.REL_WHEEL, wheel])
            events.append([t_ns, live.EV_REL, live.REL_WHEEL_HI_RES, wheel * 120])
            events.append([t_ns, live.EV_SYN, 0, 0])
        return events, decisions

    def close(self):
        pass


class LiveTestReader(threading.Thread):
    """Grab mouse, filter live through GRU -> uinput, and record raw ticks + decisions."""

    def __init__(self, device, params, thr, hold_ms, use_uinput=True):
        super().__init__(daemon=True)
        self.device = device
        self.thr = thr
        self.model = gru_np.Streaming(params)
        self.hr = live.HoldRelease(hold_ms)
        self.fd = os.open(device, os.O_RDONLY)
        self.ufd = None
        self.grabbed = False
        try:
            fcntl.ioctl(self.fd, live.EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
            if use_uinput and os.access("/dev/uinput", os.W_OK):
                fcntl.ioctl(self.fd, live.EVIOCGRAB, 1)
                fcntl.ioctl(self.fd, live.EVIOCGRAB, 0)
                self.ufd = live.make_virtual(self.fd, device)
                time.sleep(0.25)
                fcntl.ioctl(self.fd, live.EVIOCGRAB, 1)
                self.grabbed = True
        except OSError as e:
            self._cleanup_fds()
            if e.errno == errno.EBUSY:
                raise BlockingIOError("Mouse is grabbed by another process") from e
            raise
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.recording = False
        self.events = []
        self.decisions = []

    def run(self):
        frame = []
        while not self.stop_ev.is_set():
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if not r:
                continue
            try:
                data = os.read(self.fd, live.EVENT.size * 64)
            except OSError:
                break
            if not data:
                break
            for off in range(0, len(data), live.EVENT.size):
                sec, usec, typ, code, val = live.EVENT.unpack_from(data, off)
                if typ != live.EV_SYN:
                    frame.append((typ, code, val))
                    continue
                t_ns = sec * 1_000_000_000 + usec * 1000
                wheel = next((v for ty, c, v in frame if ty == live.EV_REL and c == live.REL_WHEEL), None)
                out = [e for e in frame if not (e[0] == live.EV_REL and e[1] in (live.REL_WHEEL, live.REL_WHEEL_HI_RES))]
                dec = None
                if wheel is not None:
                    p = self.model.step(t_ns, wheel)
                    emit = self.hr.feed(t_ns, wheel, p > self.thr)
                    total = sum(emit)
                    if total:
                        out += [(live.EV_REL, live.REL_WHEEL, total), (live.EV_REL, live.REL_WHEEL_HI_RES, total * 120)]
                    dec = [t_ns, wheel, round(p, 4), emit]
                if self.ufd is not None:
                    buf = b"".join(live.EVENT.pack(0, 0, ty, c, v) for ty, c, v in out)
                    os.write(self.ufd, buf + live.EVENT.pack(0, 0, live.EV_SYN, 0, 0))
                with self.lock:
                    if self.recording:
                        for ty, c, v in frame:
                            self.events.append([t_ns, ty, c, v])
                        self.events.append([t_ns, live.EV_SYN, 0, 0])
                        if dec is not None:
                            self.decisions.append(dec)
                frame = []

    def start_recording(self):
        with self.lock:
            self.events, self.decisions = [], []
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return list(self.events), list(self.decisions)

    def decision_snapshot(self):
        with self.lock:
            return list(self.decisions[-4000:])

    def _cleanup_fds(self):
        if self.grabbed:
            try:
                fcntl.ioctl(self.fd, live.EVIOCGRAB, 0)
            except OSError:
                pass
            self.grabbed = False
        if self.ufd is not None:
            try:
                fcntl.ioctl(self.ufd, live.UI_DEV_DESTROY)
            except OSError:
                pass
            try:
                os.close(self.ufd)
            except OSError:
                pass
            self.ufd = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def close(self):
        self.stop_ev.set()
        if self.is_alive():
            self.join(timeout=1.0)
        self._cleanup_fds()


def score_model_test(cues, decisions, thr, hold_ms):
    ticks = [(t, v) for t, v, _, _ in decisions]
    if not ticks:
        return {"jitter_passed": 0, "jitter_total": 0, "good_lost": 0, "good_total": 0,
                "leaks": [], "reversals": []}
    intents = dataset.sequence_intents({"cues": cues}, ticks)
    first = next((i for i, s in enumerate(intents) if s != 0), len(ticks))
    sup = [p > thr for _, _, p, _ in decisions]
    emitted = {src: et for et, _, src in policy.hold_release(ticks, sup, hold_ms)}
    jp = jt = gl = gt = 0
    leaks = []
    for i in range(first, len(ticks)):
        v = ticks[i][1]
        n = abs(v)
        if v * intents[i] < 0:
            jt += n
            if i in emitted:
                jp += n
                leaks.append((i, v, decisions[i][2]))
        else:
            gt += n
            if i not in emitted:
                gl += n
    reversals = []
    for i in range(1, len(ticks)):
        if intents[i] == intents[i - 1] or intents[i - 1] == 0:
            continue
        want_str = "up" if intents[i] > 0 else "down"
        cue = next((c for c in reversed(cues) if c["t_ns"] <= ticks[i][0] and c.get("intent") == want_str), None)
        pace = cue.get("pace", "normal") if cue else "normal"
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
        wait_ms = (ticks[j][0] - ticks[i][0]) / 1e6 if j < seg_end else float("nan")
        hr_lost = sum(abs(ticks[k][1]) for k in range(i, seg_end)
                      if ticks[k][1] * intents[i] > 0 and k not in emitted)
        reversals.append((pace, intents[i], wait_ms, eaten, hr_lost))
    return {"jitter_passed": jp, "jitter_total": jt, "good_lost": gl, "good_total": gt,
            "leaks": leaks, "reversals": reversals}


def draw_dual_strips(scr, y_raw, y_flt, decisions):
    _, w = scr.getmaxyx()
    tail = decisions[-max(w - 14, 1):]
    for idx, (_, v, _, emit) in enumerate(tail):
        x = 12 + idx
        ch, color = ("▲", 2) if v > 0 else ("▼", 1)
        try:
            scr.addstr(y_raw, x, ch, curses.color_pair(color))
            if emit:
                ev = sum(emit)
                scr.addstr(y_flt, x, "▲" if ev > 0 else "▼", curses.color_pair(2 if ev > 0 else 1))
            else:
                scr.addstr(y_flt, x, "·", curses.A_DIM)
        except curses.error:
            pass


def run_model_test(scr, device, weights=None, thr=0.7, hold_ms=100, label="", from_live_log=False):
    runner = tailer = reader = None
    log_path = None
    source = "direct"
    if not from_live_log:
        weights = weights or devices.weights_for(device)
        params = gru_np.load(weights)
        if params["w_ih"].shape[1] != dataset.IN_DIM:
            msg = f"incompatible weights {weights}: expects {params['w_ih'].shape[1]} features, need {dataset.IN_DIM}"
            scr.erase(); put(scr, 1, msg, curses.color_pair(1)); put(scr, 3, "Press any key…"); scr.refresh(); wait_key(scr)
            return msg
        try:
            reader = LiveTestReader(device, params, thr, hold_ms)
            runner = reader
        except BlockingIOError:
            from_live_log = True

    if from_live_log:
        log_path, meta = latest_live_log()
        if not log_path:
            msg = "Mouse is grabbed by another process and no live_logs/*.jsonl found — stop live.py first."
            scr.erase(); put(scr, 1, msg, curses.color_pair(1)); put(scr, 3, "Press any key…"); scr.refresh(); wait_key(scr)
            return msg
        tailer = LiveLogTailer(log_path)
        runner = tailer
        source = "live_log"
        weights = meta.get("weights") or meta.get("model") or devices.weights_for(device)
        thr = float(meta.get("thr", thr))
        hold_ms = float(meta.get("hold_ms", hold_ms))
        device = meta.get("device") or device

    plan = recorder.plan_model_test()
    wname = os.path.relpath(weights, HERE) if weights and os.path.exists(weights) else str(weights)
    mode_badge = (f"LIVE-LOG TIMESTAMP MODE ({os.path.basename(log_path)}; live.py stays running)"
                  if source == "live_log" else
                  ("DIRECT LIVE-FILTER MODE (uinput ON)" if reader.grabbed else "DIRECT READ-ONLY (no /dev/uinput)"))
    cues, started, ended = [], 0, 0
    events, decisions = [], []

    try:
        if reader is not None:
            reader.start()
        hdr = "AGAINST MODEL TEST" + (f" [{label}]" if label else "")
        scr.erase()
        put(scr, 1, hdr, curses.A_BOLD)
        put(scr, 2, f"weights: {wname}   thr={thr}   hold={hold_ms:g}ms", curses.A_DIM)
        put(scr, 3, mode_badge, curses.color_pair(2))
        put(scr, 5, "Press ENTER when your hand is on the wheel (q to cancel)", curses.A_BOLD)
        put(scr, 6, "After the ~38s sequence finishes, you'll see the audit and press k (audit), t (train), or d (discard).", curses.A_DIM)
        scr.refresh()
        while True:
            key = wait_key(scr)
            if key == ord("q"):
                return "aborted"
            if key in (10, 13, curses.KEY_ENTER):
                break
        scr.nodelay(True)
        for n in (3, 2, 1):
            scr.erase()
            put(scr, 1, hdr, curses.A_BOLD)
            put(scr, 2, f"weights: {wname}   thr={thr}   hold={hold_ms:g}ms", curses.A_DIM)
            put(scr, 3, mode_badge, curses.color_pair(2))
            put(scr, 5, f"Starts in {n}… follow the arrow & pace cues. Stop on STOP.", curses.A_BOLD)
            put(scr, 6, "After the ~38s sequence finishes, you'll see the audit and press k (audit), t (train), or d (discard).", curses.A_DIM)
            scr.refresh()
            time.sleep(1)

        started = time.monotonic_ns()
        if reader is not None:
            reader.start_recording()
        for direction, dur, pace in plan:
            t0 = time.monotonic_ns()
            cues.append({"t_ns": t0, "intent": direction, "pace": pace})
            while time.monotonic_ns() - t0 < dur * 1e9:
                if ord("q") in recorder.pending_keys(scr):
                    return "aborted"
                if source == "live_log":
                    decs = [d for d in tailer.poll() if d[0] >= started]
                else:
                    decs = reader.decision_snapshot()
                scr.erase()
                put(scr, 1, f"AGAINST MODEL TEST" + (f" [{label}]" if label else ""), curses.A_DIM)
                banner = {
                    ("up", "normal"): "▲▲▲  UP  ▲▲▲",
                    ("down", "normal"): "▼▼▼  DOWN  ▼▼▼",
                    ("up", "fast"): "▲▲▲  FAST UP (quick flick)  ▲▲▲",
                    ("down", "fast"): "▼▼▼  FAST DOWN (quick flick)  ▼▼▼",
                    ("up", "slow"): "▲  SLOW UP (1 click at a time)  ▲",
                    ("down", "slow"): "▼  SLOW DOWN (1 click at a time)  ▼",
                    ("stop", "stop"): "■  STOP — finger stays on wheel  ■",
                }[(direction, pace)]
                color = {"up": 2, "down": 1, "stop": 3}[direction]
                put(scr, 3, banner, curses.A_BOLD | curses.A_REVERSE | curses.color_pair(color))
                left = max(0.0, dur - (time.monotonic_ns() - t0) / 1e9)
                put(scr, 5, f"segment {len(cues)}/{len(plan)} ({pace})   {left:4.1f}s   (q to abort)")
                sup_n = sum(1 for _, _, _, em in decs if not em)
                rel_n = sum(1 for _, _, _, em in decs if len(em) > 1)
                note = "   (updates once live.py flushes)" if source == "live_log" and not decs else ""
                put(scr, 7, f"raw ticks: {len(decs)}   suppressed: {sup_n}   hold-released: {rel_n}{note}")
                put(scr, 9, "raw     :")
                put(scr, 10, "filtered:")
                draw_dual_strips(scr, 9, 10, decs)
                scr.refresh()
                time.sleep(0.016)
        ended = time.monotonic_ns()
        if reader is not None:
            events, decisions = reader.stop_recording()
    finally:
        runner.close()

    if source == "live_log":
        time.sleep(0.15)
        events, decisions = tailer.collect(started, ended)
        if not decisions:
            scr.erase()
            put(scr, 1, "AGAINST MODEL TEST — waiting for live log flush", curses.A_BOLD | curses.color_pair(3))
            put(scr, 3, f"No ticks flushed to {os.path.basename(log_path)} yet")
            put(scr, 4, "(an older live.py buffers output until it stops).")
            put(scr, 6, "Stop live.py (Ctrl+C) in your other terminal, then:")
            put(scr, 7, "  r  re-read log & score     q  discard")
            scr.refresh()
            while True:
                k = wait_key(scr)
                if k == ord("q"):
                    return "discarded"
                if k == ord("r"):
                    events, decisions = tailer.collect(started, ended)
                    if decisions:
                        break
                    put(scr, 9, "Still 0 ticks in window — stop live.py (Ctrl+C) first, then press r.", curses.color_pair(1))
                    scr.refresh()

    res = score_model_test(cues, decisions, thr, hold_ms)
    jp, jt, gl, gt = res["jitter_passed"], res["jitter_total"], res["good_lost"], res["good_total"]
    jp_pct = 100 * jp / max(jt, 1)
    gl_pct = 100 * gl / max(gt, 1)

    scr.erase()
    h, _ = scr.getmaxyx()
    put(scr, 1, "AGAINST MODEL TEST — Audit Summary", curses.A_BOLD)
    put(scr, 2, f"weights: {wname}   source: {source}   ticks: {len(decisions)}", curses.A_DIM)
    put(scr, 4, f"jitter passed: {jp}/{jt} ({jp_pct:.1f}%)   good lost: {gl}/{gt} ({gl_pct:.2f}%)   leaked ticks: {len(res['leaks'])}",
        curses.color_pair(2 if len(res["leaks"]) == 0 and gl == 0 else 3))
    put(scr, 5, "k  save audit-only     t  save & include in fine-tuning     d/q  discard", curses.A_BOLD | curses.A_REVERSE)
    y = 7
    max_rows = max(0, h - y - (3 if res["leaks"] else 1))
    if max_rows > 0:
        put(scr, y, "Reversals:")
        y += 1
        for pace, new_dir, wait_ms, eaten, hr_lost in res["reversals"][:max_rows]:
            stalled = hr_lost > 0 or (wait_ms == wait_ms and wait_ms > 100)
            arrow = "UP  " if new_dir > 0 else "DOWN"
            wtxt = f"{wait_ms:4.0f}ms" if wait_ms == wait_ms else " none"
            put(scr, y, f"  {pace:<6s} -> {arrow}  wait={wtxt}  eaten={eaten:2d}  hr_lost={hr_lost:2d}",
                curses.color_pair(1) if stalled else 0)
            y += 1
    if res["leaks"] and y + 1 < h:
        put(scr, y + 1, "Leaked jitter samples: " + ", ".join(f"{v:+d}(p={p:.2f})" for _, v, p in res["leaks"][:5]),
            curses.color_pair(3))
    scr.refresh()
    while True:
        key = wait_key(scr)
        if key in (ord("d"), ord("q")):
            return "discarded"
        if key in (ord("k"), ord("t")):
            include = key == ord("t")
            clean = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") if label else ""
            tag = f"against_model_{clean}" if clean else "against_model"
            extra = {
                "against_model": {
                    "weights": wname,
                    "thr": thr,
                    "hold_ms": hold_ms,
                    "source": source,
                    "live_log": os.path.basename(log_path) if log_path else None,
                    "label": label,
                },
                "include_in_train": include,
                "decisions": decisions,
            }
            path = recorder.save("model_test", events, cues, started, device, extra_meta=extra, filename_tag=tag)
            kind = "train+audit" if include else "audit-only"
            return f"saved ({kind}) {os.path.relpath(path, HERE)} | jit {jp}/{jt} ({jp_pct:.1f}%), good lost {gl}/{gt} ({gl_pct:.2f}%)"


def device_screen(scr, mouse):
    device = mouse["path"]
    reader = recorder.Reader(device)
    reader.start()
    last = ""
    while True:
        counts = session_counts(device)
        ready = all(counts.get(mode, 0) >= need for mode, need, _ in STEPS)
        profile = os.path.exists(devices.profile_path(device))
        scr.erase()
        put(scr, 1, mouse["name"], curses.A_BOLD)
        put(scr, 2, device, curses.A_DIM)
        put(scr, 3, "profile: " + ("tuned" if profile else "none (base weights)"), curses.A_DIM)
        put(scr, 5, "Record:")
        for i, (mode, need, text) in enumerate(STEPS):
            have = counts.get(mode, 0)
            mark = "✓" if have >= need else " "
            put(scr, 6 + i, f"{i + 1}  [{mark}] {text}   recorded {have}/{need}",
                curses.color_pair(2) if have >= need else 0)
        put(scr, 10, "f  fine-tune" + ("" if ready else "  (record the steps above first)"),
            0 if ready else curses.A_DIM)
        uinput_ok = os.access("/dev/uinput", os.W_OK)
        put(scr, 11, "t  test against active model (~38 s, live-filtered failure audit)")
        put(scr, 12, "l  launch live filter" + ("" if uinput_ok else "  (needs: sudo setfacl -m u:$USER:rw /dev/uinput)"),
            0 if uinput_ok else curses.A_DIM)
        put(scr, 13, "b  back    q  quit", curses.A_DIM)
        put(scr, 15, "More recordings always help: steps can be repeated, every one is kept.", curses.A_DIM)
        if last:
            put(scr, 17, f"last: {last}")
        scr.refresh()
        key = wait_key(scr)
        if key == ord("q"):
            return ("quit", None)
        if key == ord("b"):
            return ("back", None)
        if key == ord("1"):
            last = recorder.run_free(scr, reader, "down", FREE_LIMIT_S)
        elif key == ord("2"):
            last = recorder.run_free(scr, reader, "up", FREE_LIMIT_S)
        elif key == ord("3"):
            last = recorder.run_sequence(scr, reader, "sequence", recorder.plan_alternating())
        elif key == ord("t"):
            last = run_model_test(scr, device)
        elif key == ord("f") and ready:
            last = run_finetune(scr, device)
        elif key == ord("l") and uinput_ok:
            return ("launch", device)


def main(scr, args):
    init_colors()
    if args.against_model:
        device = args.device
        if not device and args.from_live_log:
            _, meta = latest_live_log()
            if meta:
                device = meta.get("device")
        if not device:
            mice = [m for m in devices.wheel_mice() if m["readable"]]
            if len(mice) == 1:
                device = mice[0]["path"]
            else:
                mouse = pick_device(scr)
                if not mouse:
                    return ("quit", None)
                device = mouse["path"]
        res = run_model_test(scr, device, weights=args.weights, thr=args.thr, hold_ms=args.hold_ms,
                             label=args.label, from_live_log=args.from_live_log)
        return ("test_done", res)
    while True:
        mouse = pick_device(scr)
        if not mouse:
            return ("quit", None)
        action, device = device_screen(scr, mouse)
        if action == "quit":
            return ("quit", None)
        if action == "launch":
            return ("launch", device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MouseScrollFixer calibration and closed-loop model testing.")
    ap.add_argument("--against-model", action="store_true", help="run closed-loop cued test against the active model")
    ap.add_argument("--from-live-log", action="store_true", help="align cues to the active live_logs/*.jsonl by timestamp")
    ap.add_argument("--label", default="", help="label stamped into the saved against_model filename and metadata")
    ap.add_argument("--device", help="input device path (auto-detected if omitted)")
    ap.add_argument("--weights", help="weights .npz path (default: device profile else weights/base.npz)")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--hold-ms", type=float, default=100)
    cli_args = ap.parse_args()
    action, payload = curses.wrapper(main, cli_args)
    if action == "launch":
        live_py = os.path.join(HERE, "live.py")
        os.execv(sys.executable, [sys.executable, live_py, "--device", payload, "--quiet"])
    if action == "test_done" and payload:
        print(payload)
