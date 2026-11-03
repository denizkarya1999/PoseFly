#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

# Explicit override > activated environment > repository environment > system.
if [[ -n "${POSEFLY_PYTHON:-}" ]]; then
    PYTHON="$POSEFLY_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON=python3
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python not found. Install Python 3.10+ and follow README.md." >&2
    exit 1
fi
"$PYTHON" - <<'PY'
import importlib.util
import sys
if sys.version_info < (3, 10):
    sys.exit("PoseFly needs Python 3.10 or newer.")
missing = [name for name in ('tkinter', 'cv2', 'numpy', 'PIL', 'matplotlib', 'ultralytics')
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Missing dependencies: " + ', '.join(missing) + ". Follow the Linux setup in README.md.")
PY
if [[ -z "${DISPLAY:-}" ]]; then
    echo "PoseFly needs a graphical session with DISPLAY set (X11 or XWayland)." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
# Register only the taskbar identity/icon; NoDisplay keeps it out of app menus.
# A read-only home directory must not prevent the camera application from opening.
"$PYTHON" "$PROJECT_DIR/linux/install_desktop.py" --quiet || true
exec "$PYTHON" "$SCRIPT_DIR/posefly_gui.py" "$@"
