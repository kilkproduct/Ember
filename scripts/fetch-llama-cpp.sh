#!/usr/bin/env bash
# Fetch the pinned llama.cpp source tree used by Ember's native inference engine.
#
#   scripts/fetch-llama-cpp.sh [--dir PATH] [--force]
#
# Downloads the release tarball recorded in third_party/llama.cpp.version from
# codeload.github.com, verifies its SHA-256, and extracts it. Prebuilt binaries
# are never downloaded. The destination is git-ignored: CI and local builds both
# fetch the same pinned revision.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/third_party/llama.cpp.version"
DEST="${REPO_ROOT}/third_party/llama.cpp"
CACHE_DIR="${EMBER_LLAMA_CACHE:-${REPO_ROOT}/third_party/.cache}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DEST="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "usage: $0 [--dir PATH] [--force]" >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

read_manifest() {
  python3 - "$MANIFEST" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
}

TAG="$(read_manifest tag)"
URL="$(read_manifest tarball_url)"
SHA="$(read_manifest sha256)"
STAMP="${DEST}/.ember-fetch-stamp"

if [[ $FORCE -eq 0 && -f "$STAMP" && "$(cat "$STAMP")" == "${TAG}:${SHA}" && -f "${DEST}/CMakeLists.txt" ]]; then
  echo "llama.cpp ${TAG} already present at ${DEST}"
  exit 0
fi

mkdir -p "$CACHE_DIR"
TARBALL="${CACHE_DIR}/llama.cpp-${TAG}.tar.gz"

if [[ ! -f "$TARBALL" ]]; then
  echo "Downloading llama.cpp ${TAG} from ${URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --max-time 600 -o "$TARBALL.part" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$TARBALL.part" "$URL"
  else
    echo "need curl or wget to fetch llama.cpp" >&2
    exit 1
  fi
  mv "$TARBALL.part" "$TARBALL"
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$TARBALL" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
else
  echo "need sha256sum or shasum to verify the download" >&2
  exit 1
fi

if [[ "$ACTUAL" != "$SHA" ]]; then
  echo "SHA-256 mismatch for llama.cpp ${TAG}:" >&2
  echo "  expected ${SHA}" >&2
  echo "  actual   ${ACTUAL}" >&2
  rm -f "$TARBALL"
  exit 1
fi
echo "SHA-256 verified: ${SHA}"

rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$TARBALL" -C "$DEST" --strip-components=1
echo "${TAG}:${SHA}" > "$STAMP"
echo "llama.cpp ${TAG} extracted to ${DEST}"
