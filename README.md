# MouseScrollFixer

Experiment: suppress mouse-wheel jitter (ticks in the wrong direction, worst at the end of a scroll)
in real time, ideally with a very small model (< 1M params, likely < 10k) sitting between the
device and the compositor.

Mouse: `YICHIP 2.4G Receiver Mouse` (USB 3515:3020), read via evdev.

## Scripts

| Script | What it does |
|---|---|
| `recorder.py` | Terminal UI that records raw evdev events, labelled by intent: `d` down, `u` up, `s` cued up/down sequence, `r` rest. Writes `sessions/<time>_<mode>.jsonl`. |
| `analyze.py` | Per-session summary: jitter rate, bursts ending in a reversal, reversal delays, reaction times. |
| `baseline.py` | Rule-based filter (hold opposite ticks, release if ≥ M notches within W ms) scored on the sessions. The bar a model has to beat. |

```sh
python3 recorder.py
python3 analyze.py            # all sessions, or pass paths
python3 baseline.py
```

## Device access

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
- Stdlib only for recording and analysis, so it runs anywhere without a venv.
