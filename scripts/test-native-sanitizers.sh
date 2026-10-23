#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the already checksum-verified upstream source. No device result implied.
BUILD_DIR="${EMBER_SANITIZER_BUILD_DIR:-native/build-llama-sanitized}"
cmake -S native/llama -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DEMBER_BUILD_LLAMA_TESTS=ON \
  -DCMAKE_C_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build "$BUILD_DIR" -j2
TMP="$(mktemp -d)";trap 'rm -rf "$TMP"' EXIT
python scripts/make_test_gguf.py "$TMP/f32.gguf"
printf 'not a GGUF model' > "$TMP/invalid.emb"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$BUILD_DIR/test_llama_engine" "$TMP/f32.gguf" "$TMP/invalid.emb"
