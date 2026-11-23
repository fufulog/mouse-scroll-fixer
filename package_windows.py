#!/usr/bin/env python3
"""Build dist/MouseScrollFixer-windows.zip: everything the Windows filter needs, nothing else."""
import os, zipfile, glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dist", "MouseScrollFixer-windows.zip")
ROOT = "MouseScrollFixer"
FILES = ["live_windows.py", "dataset.py", "policy.py"]

START_BAT = r"""@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Python is not installed. Get it from https://www.python.org/downloads/ & pause & exit /b 1)
py live_windows.py --quiet
pause
"""

TEST_BAT = r"""@echo off
cd /d "%~dp0"
py live_windows.py --dry-run
pause
"""

README = """MouseScrollFixer for Windows
============================

1. Install Python 3 from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
2. Double-click start.bat. Scroll as normal. Close the window to stop.

test.bat shows what the filter would do without changing anything.
Run at login: press Win+R, type shell:startup, put a shortcut to start.bat in that folder.

NOTE: this Windows version is untested on real Windows. If scrolling acts up, close the
window; the filter stops immediately. Logs are written to the live_logs folder.
Known limit: scrolling inside programs run as administrator may lose some ticks.
"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(HERE, f), f"{ROOT}/{f}")
        for p in glob.glob(os.path.join(HERE, "windows", "profiles", "*.json")):
            z.write(p, f"{ROOT}/windows/profiles/{os.path.basename(p)}")
        z.writestr(f"{ROOT}/start.bat", START_BAT.replace("\n", "\r\n"))
        z.writestr(f"{ROOT}/test.bat", TEST_BAT.replace("\n", "\r\n"))
        z.writestr(f"{ROOT}/README.txt", README.replace("\n", "\r\n"))
    print(OUT)


if __name__ == "__main__":
    main()
