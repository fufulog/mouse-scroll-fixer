#!/usr/bin/env python3
"""MouseScrollFixer: calibrate the jitter filter to a mouse and run it.

  pick mouse -> record (down, up, cued sequence) -> fine-tune -> compare -> save profile -> launch

Python 3 + NumPy only. Recordings land in sessions/, profiles in profiles/.
"""
import curses, glob, json, os, sys

import devices, finetune, gru_np, recorder

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
        put(scr, 11, "l  launch live filter" + ("" if uinput_ok else "  (needs: sudo setfacl -m u:$USER:rw /dev/uinput)"),
            0 if uinput_ok else curses.A_DIM)
        put(scr, 12, "b  back    q  quit", curses.A_DIM)
        put(scr, 14, "More recordings always help: steps can be repeated, every one is kept.", curses.A_DIM)
        if last:
            put(scr, 16, f"last: {last}")
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
        elif key == ord("f") and ready:
            last = run_finetune(scr, device)
        elif key == ord("l") and uinput_ok:
            return ("launch", device)


def main(scr):
    init_colors()
    while True:
        mouse = pick_device(scr)
        if not mouse:
            return None
        action, device = device_screen(scr, mouse)
        if action == "quit":
            return None
        if action == "launch":
            return device


if __name__ == "__main__":
    device = curses.wrapper(main)
    if device:
        live = os.path.join(HERE, "live.py")
        os.execv(sys.executable, [sys.executable, live, "--device", device, "--quiet"])
