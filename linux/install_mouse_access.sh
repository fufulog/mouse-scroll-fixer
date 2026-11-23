#!/usr/bin/env bash
# Give USER read-only access to every mouse event device, re-applied by udev on every plug-in.
# Keyboards (and combo devices that are also keyboards) are excluded. /dev/uinput is NOT touched.
#   sudo linux/install_mouse_access.sh [user]      (default: the user who ran sudo)
#   sudo rm /etc/udev/rules.d/72-mousescrollfixer.rules   to undo
set -euo pipefail
user="${1:-${SUDO_USER:-}}"
[[ -n "$user" && "$user" != root ]] || { echo "usage: sudo $0 <user>" >&2; exit 1; }
id "$user" >/dev/null
rule=/etc/udev/rules.d/72-mousescrollfixer.rules
cat > "$rule" <<RULE
# MouseScrollFixer: read-only ACL for $user on mouse event nodes (not keyboards).
ACTION=="add|change", SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_MOUSE}=="1", ENV{ID_INPUT_KEYBOARD}!="1", RUN+="/usr/bin/setfacl -m u:$user:r \$env{DEVNAME}"
RULE
udevadm control --reload
udevadm trigger --subsystem-match=input --action=change
udevadm settle
echo "installed $rule"
