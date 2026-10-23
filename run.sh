#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m ember.server --host 0.0.0.0 --port 8000
