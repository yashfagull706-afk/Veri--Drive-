#!/usr/bin/env bash
# Veri-Drive launcher for Linux / macOS (mirrors start_veri_drive.bat).
# Usage:  ./start_veri_drive.sh
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  Starting Veri-Drive Gate Security System..."
echo "============================================================"

PYTHON_BIN="python3"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

# Open the dashboard once the server is ready (background, best-effort)
(
  sleep 12
  if command -v xdg-open >/dev/null 2>&1; then xdg-open http://127.0.0.1:5000
  elif command -v open >/dev/null 2>&1; then open http://127.0.0.1:5000
  fi
) &

exec "$PYTHON_BIN" veri_drive.py
