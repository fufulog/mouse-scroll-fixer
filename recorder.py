#!/usr/bin/env python3
"""Record raw evdev wheel data from the mouse, labelled by intended direction.

Modes:  d = scroll DOWN session, u = scroll UP session,
        s = cued sequence (UP/DOWN alternating, random durations),
        r = rest (hand on wheel, don't scroll) to catch spurious ticks.
Down/up/rest: ENTER arms; recording starts at the first wheel event; ENTER stops.
Every raw event (wheel, hi-res wheel, X/Y motion, buttons, SYN) is saved with
its kernel CLOCK_MONOTONIC timestamp to sessions/<time>_<mode>.jsonl.
"""
import curses, fcntl, json, os, random, struct, sys, threading, time

DEVICE = "/dev/input/by-id/usb-YICHIP_2.4G_Receiver_b120300001-if01-event-mouse"
EVENT = struct.Struct("llHHi")  # timeval sec, usec, type, code, value
EVIOCSCLOCKID = 0x400445A0
EV_REL, REL_WHEEL, REL_WHEEL_HI_RES = 0x02, 0x08, 0x0B
WHEEL_CODES = {REL_WHEEL, REL_WHEEL_HI_RES}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
SEQ_SEGMENTS, SEQ_MIN_S, SEQ_MAX_S = 10, 2.5, 4.5


class Reader(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.fd = os.open(path, os.O_RDONLY)
        fcntl.ioctl(self.fd, EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
        self.lock = threading.Lock()
        self.events = []  # [t_ns, type, code, value]
        self.recording = False
        self.armed = False

    def run(self):
        while True:
            data = os.read(self.fd, EVENT.size * 64)
            for off in range(0, len(data), EVENT.size):
                sec, usec, typ, code, value = EVENT.unpack_from(data, off)
                ev = [sec * 1_000_000_000 + usec * 1000, typ, code, value]
                is_wheel = typ == EV_REL and code in WHEEL_CODES
                with self.lock:
                    if self.armed and is_wheel:
                        self.armed, self.recording = False, True
                    if self.recording:
                        self.events.append(ev)

    def start_session(self, armed):
        with self.lock:
            self.events = []
            self.armed, self.recording = armed, not armed

    def stop_session(self):
        with self.lock:
            self.armed = self.recording = False
            return self.events

    def snapshot(self):
        with self.lock:
            return self.recording, self.armed, list(self.events[-4000:])


def save(mode, events, cues, started_ns):
    os.makedirs(OUT_DIR, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + f"_{mode}.jsonl"
    path = os.path.join(OUT_DIR, name)
    meta = {"mode": mode, "device": DEVICE, "clock": "CLOCK_MONOTONIC",
            "started_ns": started_ns, "cues": cues, "n_events": len(events),
            "sign": "REL_WHEEL > 0 means UP (away from user)"}
    with open(path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def hires_values(events):
    return [v for _, t, c, v in events if t == EV_REL and c == REL_WHEEL_HI_RES]


def draw_strip(win, y, values, width):
    """One char per hi-res event, newest on the right: green=up, red=down."""
    vals = values[-width:]
    for i, v in enumerate(vals):
        ch, color = ("▲", 2) if v > 0 else ("▼", 1)
        try:
            win.addstr(y, i + 2, ch, curses.color_pair(color))
        except curses.error:
            pass


def draw_stats(win, y, values, intent):
    up = sum(1 for v in values if v > 0)
    down = sum(1 for v in values if v < 0)
    line = f"hi-res events: {len(values)}   up: {up}   down: {down}"
    if intent in ("down", "up") and values:
        wrong = up if intent == "down" else down
        line += f"   against intent: {wrong} ({100 * wrong / len(values):.1f}%)"
    win.addstr(y, 2, line)


def pending_keys(scr):
    """Drain every queued key. Scrolling over the terminal floods the input queue with
    arrow-key/mouse escape sequences; reading one key per frame left Enter/q stuck behind them."""
    keys = set()
    while (key := scr.getch()) != -1:
        keys.add(key)
    return keys


def run_free(scr, reader, mode):
    reader.start_session(armed=True)
    started = time.monotonic_ns()
    scr.nodelay(True)
    while True:
        keys = pending_keys(scr)
        if keys & {10, 13, curses.KEY_ENTER}:
            break
        if ord("q") in keys:
            reader.stop_session()
            return "discarded"
        recording, armed, evs = reader.snapshot()
        scr.erase()
        h, w = scr.getmaxyx()
        title = {"down": "SCROLL DOWN ▼", "up": "SCROLL UP ▲", "rest": "REST: finger on wheel, don't scroll"}[mode]
        scr.addstr(1, 2, title, curses.A_BOLD | curses.A_REVERSE)
        state = "waiting for first wheel event…" if armed else "RECORDING — ENTER to stop & save, q to discard"
        if mode == "rest":
            state = "RECORDING — ENTER to stop & save, q to discard"
            if armed:
                reader.start_session(armed=False)
        scr.addstr(3, 2, state)
        if mode != "rest":
            scr.addstr(4, 2, "scroll in bursts, pause whenever you like", curses.A_DIM)
        vals = hires_values(evs)
        draw_stats(scr, 5, vals, mode)
        draw_strip(scr, 7, vals, w - 4)
        scr.refresh()
        time.sleep(0.016)
    events = reader.stop_session()
    return save(mode, events, [], started)


def plan_alternating():
    dirs = ["up", "down"] * (SEQ_SEGMENTS // 2)
    return [(d, random.uniform(SEQ_MIN_S, SEQ_MAX_S)) for d in dirs] + [("stop", 1.5)]


def run_sequence(scr, reader, mode, plan):
    scr.nodelay(True)
    for n in (3, 2, 1):
        scr.erase()
        scr.addstr(2, 2, f"Sequence starts in {n}… follow the arrow. Stop scrolling when it says STOP.", curses.A_BOLD)
        scr.refresh()
        time.sleep(1)
    reader.start_session(armed=False)
    started = time.monotonic_ns()
    cues = []
    for direction, dur in plan:
        t0 = time.monotonic_ns()
        cues.append({"t_ns": t0, "intent": direction})
        while time.monotonic_ns() - t0 < dur * 1e9:
            if ord("q") in pending_keys(scr):
                reader.stop_session()
                return "discarded"
            _, _, evs = reader.snapshot()
            scr.erase()
            h, w = scr.getmaxyx()
            label = {"up": "▲▲▲  UP  ▲▲▲", "down": "▼▼▼  DOWN  ▼▼▼",
                     "stop": "■  STOP — finger stays on wheel  ■"}[direction]
            color = {"up": 2, "down": 1, "stop": 3}[direction]
            scr.addstr(2, 2, label, curses.A_BOLD | curses.A_REVERSE | curses.color_pair(color))
            left = dur - (time.monotonic_ns() - t0) / 1e9
            scr.addstr(4, 2, f"segment {len(cues)}/{len(plan)}   {left:4.1f}s   (q to abort)")
            vals = hires_values(evs)
            draw_stats(scr, 6, vals, None)
            draw_strip(scr, 8, vals, w - 4)
            scr.refresh()
            time.sleep(0.008)
    events = reader.stop_session()
    return save(mode, events, cues, started)


def main(scr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    reader = Reader(DEVICE)
    reader.start()
    last = ""
    while True:
        scr.nodelay(True)
        pending_keys(scr)  # discard wheel junk queued during the last session
        scr.nodelay(False)
        scr.erase()
        scr.addstr(1, 2, "Wheel jitter recorder", curses.A_BOLD)
        scr.addstr(3, 2, "d  scroll DOWN session")
        scr.addstr(4, 2, "u  scroll UP session")
        scr.addstr(5, 2, "s  cued sequence (UP/DOWN alternating, ~35s)")
        scr.addstr(6, 2, "r  rest session (finger on wheel, no scrolling)")
        scr.addstr(7, 2, "q  quit")
        scr.addstr(9, 2, "In d/u: scroll in bursts and pause whenever you like; pauses are found from the data.", curses.A_DIM)
        if last:
            scr.addstr(11, 2, f"last: {last}")
        scr.refresh()
        key = scr.getch()
        if key == ord("q"):
            return
        modes = {ord("d"): "down", ord("u"): "up", ord("r"): "rest"}
        if key in modes:
            last = run_free(scr, reader, modes[key])
        elif key == ord("s"):
            last = run_sequence(scr, reader, "sequence", plan_alternating())


if __name__ == "__main__":
    if not os.access(DEVICE, os.R_OK):
        sys.exit(f"No read access to {DEVICE}.\n"
                 f"Grant it (until replug/reboot): sudo setfacl -m u:$USER:r {os.path.realpath(DEVICE)}")
    curses.wrapper(main)
