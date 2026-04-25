#!/usr/bin/env bash
# Start the FastAPI backend + the Vite dev server for the React frontend.
# Press Ctrl-C once to stop both.
#
# Layout:
#   repo/
#   ├── back/main.py        FastAPI
#   ├── models/             notebooks, scripts, src, outputs (saved models)
#   └── front/              React + Vite SPA
#
# First time you run this:
#   pip install -r models/requirements.txt
#   (cd front && npm install)
#
# Then:
#   ./run.sh

set -euo pipefail

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT"

# ---------- preflight ----------
if [[ ! -d "front/node_modules" ]]; then
  echo "front/node_modules missing — running 'npm install' first…"
  (cd front && npm install)
fi

if ! python3 -c "import fastapi, uvicorn, xgboost" >/dev/null 2>&1; then
  echo "python deps missing — run 'pip install -r models/requirements.txt' first."
  exit 1
fi

# ---------- start ----------
echo "▶ backend   : http://localhost:8000   (uvicorn back.main:app)"
echo "▶ frontend  : http://localhost:5173   (vite dev)"
echo

# Trap so killing this script cleans up both children.
pids=()
cleanup() {
  echo
  echo "stopping…"
  for pid in "${pids[@]}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Backend (PYTHONPATH=models so `from src.X import Y` resolves)
PYTHONPATH="$ROOT/models" \
  python3 -m uvicorn back.main:app --host 0.0.0.0 --port 8000 --reload \
    --reload-dir back --reload-dir models/src &
pids+=("$!")

# Frontend
(cd front && npm run dev -- --host 0.0.0.0) &
pids+=("$!")

wait
