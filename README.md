# MouseScrollFixer

Experiment: suppress mouse-wheel jitter (ticks in the wrong direction, worst at the end of a scroll)
in real time, ideally with a very small model (< 1M params, likely < 10k) sitting between the
device and the compositor.

Mice so far: `YICHIP 2.4G Receiver Mouse` (USB 3515:3020) and a generic `USB Receiver` mouse, read via evdev.

## Status

| Part | State |
|---|---|
| Linux live filter (`live.py`) | Works; used daily, "basically perfect" on both mice |
| Calibration app (`app.py`) | Works; per-mouse fine-tune in seconds |
| Windows filter (`live_windows.py`, zip) | **Untested on real Windows.** Written and checked on Linux only (imports, model output matches the NumPy GRU). Expect first-run bugs. |

## App

```sh
uv run python app.py        # needs only Python 3 + NumPy
```

Pick a mouse, record the three steps (down, up, 2× cued sequence; about 2.5 minutes), fine-tune
(seconds, starting from `weights/base.npz`), compare base vs tuned on held-out data, save the profile
to `profiles/<device>.npz`, and launch the live filter with it. `live.py` loads a device's profile
automatically and falls back to the base weights.

## Windows (untested)

> Not yet run on a real Windows machine. The hook, `SendInput` and the batch files have only been
> checked on Linux. If it misbehaves, close the window: Windows removes the hook when the process exits.

Calibrate on Linux, run on Windows. Normal users get one zip:

```sh
uv run python export_windows.py profiles/<device>.npz   # after a new fine-tune; JSON lands in windows/profiles/
python3 package_windows.py                              # -> dist/MouseScrollFixer-windows.zip
```

On the Windows machine:

1. Install Python 3 from python.org (tick "Add python.exe to PATH"). No NumPy, driver or admin needed.
2. Unzip, double-click `start.bat`. Close the window to stop.
3. `test.bat` = dry run: prints what it would suppress, changes nothing.
4. Run at login: Win+R → `shell:startup` → put a shortcut to `start.bat` there.

It uses the most recently trained profile in `windows/profiles/`; pick another with
`py live_windows.py --weights windows\profiles\<name>.json`. Logs go to `live_logs\win-*.jsonl`.

How it works: a low-level mouse hook (`WH_MOUSE_LL`) replaces evdev grab + uinput. Passed ticks
continue untouched; suppressed ticks are swallowed; released ticks are re-sent with `SendInput`
(tagged so the hook ignores its own events).

Known limits:

- The hook sees all mice merged, so it filters every whole-notch wheel event with one profile.
  Precision touchpads and true hi-res wheels send partial notches and pass untouched.
- Released ticks can't reach elevated (admin) windows (UIPI). Run it elevated if that matters.
- Windows drops a hook whose callback is too slow; one model step is ~100 µs, far under the limit.
- Timestamps come from the hook, not the device, so gaps can be a few ms off versus Linux training data.

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
| `live_windows.py` | Windows live filter (untested): low-level mouse hook, pure-Python GRU |
| `export_windows.py` | `.npz` profile → `windows/profiles/*.json` for Windows |
| `package_windows.py` | Builds `dist/MouseScrollFixer-windows.zip` (filter, profiles, `start.bat`, `test.bat`, `README.txt`) |
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
