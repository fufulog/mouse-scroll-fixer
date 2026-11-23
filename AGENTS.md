# AGENTS.md

Instructions for coding agents working in this repo. Read `README.md` for what each script does and
`WIKI.md` for invariants, gotchas and history before changing code.

## Working rules

- Sessions in `sessions/` are the dataset: commit them, never rewrite or delete them. Re-label in code (`dataset.py`).
- Dependencies: stdlib only for recording and analysis; NumPy only for the app, fine-tuning and `live.py`.
  `live_windows.py` must stay stock-Python (no NumPy). Torch lives only in the `research` uv group.
- `weights/base.npz` and `windows/profiles/*.json` are committed. `profiles/`, `models/`, `live_logs/`, `dist/` are per-machine and gitignored.
- The Windows filter is **untested on real Windows**. Don't claim it works; keep the untested notes in
  `README.md` and `package_windows.py` (zip README.txt) until it has run on a real machine.
- If `live_windows.py` gains an import, add that file to `FILES` in `package_windows.py`, then rebuild with `python3 package_windows.py`.
- Anything for another computer or a non-technical user ships as one zip with a double-click launcher and a short README.txt.
- Device access: never add the user to the `input` group. Mice get a read-only ACL via `linux/install_mouse_access.sh`;
  `/dev/uinput` stays manual on purpose (it can inject keystrokes).
- Commits: make local commits as you work, one per finished piece. Never push unless asked.

## Wiki (long-term memory)

The repo-local skills maintain `WIKI.md`. They are plain copies in both `.agents/skills/` and `.claude/skills/` (no symlinks); keep the two in sync:

- `write-wiki`: create a `WIKI.md` for a folder that has none, when work there taught something non-obvious
  (a bug with a hidden cause, a new subsystem, rules that must hold).
- `update-wiki`: after each commit that touches a folder with a `WIKI.md`, add what was learned
  (fixes, invariants, gotchas), with a source citation (thread id, commit hash or `code`).

Commit wiki changes on their own, with only `WIKI.md` files in the commit.
