#!/usr/bin/env bash
set -euo pipefail

# Run_PoseFly_Linux.sh
# Launch PoseFly Tkinter GUI on Linux using python3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: ensure GUI can find assets relative to this folder
export PYTHONUNBUFFERED=1

# Prefer python3
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install it with: sudo apt install python3" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/posefly_gui.py"