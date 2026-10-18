#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x /tmp/ember-venv/bin/python ]]; then
  PY=/tmp/ember-venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi
exec "$PY" -m uvicorn ember.server:app --host 0.0.0.0 --port 8000
