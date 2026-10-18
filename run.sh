#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/web"
# Static app — no FastAPI/pydantic needed (those fail to build on Termux/Android).
exec python3 -m http.server 8000 --bind 0.0.0.0
