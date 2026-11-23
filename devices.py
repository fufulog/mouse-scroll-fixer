"""Find mice with a scroll wheel, a stable path for each, and its profile file."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "profiles")
BASE_WEIGHTS = os.path.join(HERE, "weights", "base.npz")
REL_WHEEL = 0x08


def _stable_links():
    """eventN -> /dev/input/by-id (preferred) or by-path symlink."""
    links = {}
    for folder in ("by-path", "by-id"):  # by-id last so it wins
        d = f"/dev/input/{folder}"
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                target = os.path.basename(os.path.realpath(os.path.join(d, name)))
                if target.startswith("event"):
                    links[target] = os.path.join(d, name)
    return links


def wheel_mice():
    """[{name, event, path, readable}] for every input device that has REL_WHEEL."""
    links, found = _stable_links(), []
    with open("/proc/bus/input/devices") as f:
        blocks = f.read().strip().split("\n\n")
    for block in blocks:
        name = re.search(r'N: Name="(.*)"', block)
        handlers = re.search(r"H: Handlers=(.*)", block)
        rel = re.search(r"B: REL=(\w+)", block)
        if not (name and handlers and rel) or not int(rel.group(1), 16) >> REL_WHEEL & 1:
            continue
        if "MouseScrollFixer" in name.group(1):
            continue
        event = next((h for h in handlers.group(1).split() if h.startswith("event")), None)
        if not event:
            continue
        path = links.get(event, f"/dev/input/{event}")
        found.append({"name": name.group(1), "event": event, "path": path,
                      "readable": os.access(path, os.R_OK)})
    return found


def slug(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(path))


def profile_path(device_path):
    return os.path.join(PROFILES, slug(device_path) + ".npz")


def weights_for(device_path):
    p = profile_path(device_path)
    return p if os.path.exists(p) else BASE_WEIGHTS


def access_hint(path):
    return (f"sudo setfacl -m u:$USER:r {os.path.realpath(path)}   (this plug-in only)\n"
            f"  permanent, every mouse: sudo {os.path.join(HERE, 'linux', 'install_mouse_access.sh')}")
