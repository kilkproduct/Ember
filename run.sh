#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Build the host C++ inference engine first so `gen.run` executes in compiled
# native code instead of falling back to the NumPy reference generator.
# Android builds this through the NDK/CMake path in android/app/build.gradle.
if command -v cmake >/dev/null 2>&1; then
  cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build native/build -j >/dev/null
elif command -v c++ >/dev/null 2>&1; then
  mkdir -p native/build
  c++ -std=c++17 -O3 -fno-fast-math -fPIC -shared \
    -Inative/include -Inative/src \
    native/src/bpe.cpp native/src/model.cpp native/src/engine.cpp native/src/jni_bridge.cpp \
    -pthread -o native/build/libember_infer.so
else
  echo "warning: no cmake or c++ toolchain found; native C++ inference will be unavailable" >&2
fi

HOST="${EMBER_HOST:-0.0.0.0}"
PORT="${EMBER_PORT:-8000}"
exec python3 -m ember.server --host "$HOST" --port "$PORT" "$@"
