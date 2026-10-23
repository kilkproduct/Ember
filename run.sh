#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./scripts/build-native.sh
export EMBER_LLAMA_LIB="$PWD/native/build-llama/libember_llama.so"
exec python3 -m ember.server --host "${EMBER_HOST:-0.0.0.0}" --port "${EMBER_PORT:-8000}" "$@"
