#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/fetch-llama-cpp.sh
cmake -S native/llama -B native/build-llama -DCMAKE_BUILD_TYPE=Release -DEMBER_BUILD_LLAMA_TESTS=ON
cmake --build native/build-llama -j "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
