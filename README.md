# MouseScrollFixer

Experiment: suppress mouse-wheel jitter (ticks in the wrong direction, worst at the end of a scroll)
in real time, ideally with a very small model (< 1M params, likely < 10k) sitting between the
device and the compositor.

Mouse: `YICHIP 2.4G Receiver Mouse` (USB 3515:3020), read via evdev.

## App

```sh
uv run python app.py        # needs only Python 3 + NumPy
```

Pick a mouse, record the three steps (down, up, 2× cued sequence; about 2.5 minutes), fine-tune
(seconds, starting from `weights/base.npz`), compare base vs tuned on held-out data, save the profile
to `profiles/<device>.npz`, and launch the live filter with it. `live.py` loads a device's profile
automatically and falls back to the base weights.

## Windows

Calibrate on Linux, run on Windows. `live_windows.py` needs only a stock Python 3 (python.org or
`winget install Python.Python.3.13`): no NumPy, no driver, no admin.

1. On Linux, export a profile (already done for both mice; the JSON files are committed):
   `uv run python export_windows.py profiles/<device>.npz`
2. On Windows, get the repo (`git clone`, or copy the folder) and run in a terminal:
   ```bat
   py live_windows.py --dry-run     REM decides and prints, changes nothing
   py live_windows.py               REM filters for real; Ctrl+C stops
   ```
   It defaults to the most recently trained profile in `windows/profiles/`; pick another with
   `--weights windows\profiles\<name>.json`.
3. Start it at login (optional): Task Scheduler → Create Task → trigger "At log on" → action
   `pythonw.exe` with arguments `live_windows.py --quiet` and "Start in" set to the repo folder.

How it differs from Linux: a low-level mouse hook (`WH_MOUSE_LL`) replaces evdev grab + uinput. Passed
ticks continue untouched; suppressed ticks are swallowed; released ticks are re-sent with
`SendInput`. The hook sees all mice merged, so it filters every whole-notch wheel event (precision
touchpads and true hi-res wheels send partial notches and pass through untouched).

## Scripts

| Script | What it does |
|---|---|
| `recorder.py` | Terminal UI that records raw evdev events, labelled by intent: `d` down, `u` up, `s` cued up/down sequence, `r` rest. Writes `sessions/<time>_<mode>.jsonl`. |
| `analyze.py` | Per-session summary: jitter rate, bursts ending in a reversal, reversal delays, reaction times. |
| `app.py` | Calibration TUI: record → fine-tune → save profile → launch |
| `finetune.py` | NumPy fine-tuning for one device (CLI too: `python finetune.py <device path> --out profiles/x.npz`) |
| `gru_np.py` | NumPy GRU: streaming step, batched forward/backward (checked against torch autograd), Adam |
| `live.py` | Live filter: grabs the mouse, re-emits through a uinput virtual mouse |
| `devices.py` | Finds wheel mice, stable paths, profile paths |
| `train.py` / `evaluate.py` | Research: torch training of the base model. `uv run --group research python train.py` |
| `baseline.py` | Rule-based filter (hold opposite ticks, release if ≥ M notches within W ms) scored on the sessions. The bar a model has to beat. |

```sh
python3 recorder.py
python3 analyze.py            # all sessions, or pass paths
python3 baseline.py
```

## Device access

Permanent, for every mouse you plug in (read-only ACL via a udev rule; keyboards excluded):
`sudo linux/install_mouse_access.sh`. Undo: `sudo rm /etc/udev/rules.d/72-mousescrollfixer.rules`.
`/dev/uinput` (needed by `live.py`) is deliberately left manual and resets each reboot, because it
allows injecting keystrokes: `sudo setfacl -m u:$USER:rw /dev/uinput`.

### Old one-off method

Your user is not in the `input` group (on purpose: that group can read the keyboard too).
Grant read access to just the mouse node; it resets on replug or reboot:

```sh
sudo setfacl -m u:$USER:r "$(realpath /dev/input/by-id/usb-YICHIP_2.4G_Receiver_b120300001-if01-event-mouse)"
```

## Session format

JSONL. Line 1 is metadata (`mode`, `cues` with `t_ns` + `intent`, clock). Every other line is one
raw event `[t_ns, type, code, value]`, timestamped by the kernel on `CLOCK_MONOTONIC`; cue times use
the same clock. `REL_WHEEL` (type 2, code 8) > 0 means up.

## Working rules

- Sessions are the dataset: commit them, never rewrite them. Re-label in the analysis code instead.
- Stdlib only for recording and analysis. App, fine-tuning and live filter: NumPy only. Torch stays in the `research` group.
- `weights/base.npz` is committed (the starting point for every fine-tune). `profiles/`, `models/`, `live_logs/` are per-machine and gitignored.
