# Ember — On-Device Small Language Foundry (v3)

A **small language model you train and run directly on your own Android device**.

- **Frontend**: Native Kotlin + Jetpack Compose (`android/`), plus an optional desktop web companion (`web/`).
- **Training & Data Backend (`ember/`)**: Written completely in **Python** (packaged into the Android APK via **Chaquopy 17.0** with CPython 3.13, `numpy`, and `zstandard`) — SQLite WAL corpus, multi-format dataset ingestion (Parquet plain/snappy/gzip/zstd, JSONL/NDJSON, CSV/TSV, HTML, XML/RSS, Markdown, UTF-8/UTF-16, ZIP/TAR/GZ/BZ2/XZ/ZST with bomb & traversal guards), byte-level BPE training, document-masked GPT forward/backward passes, AdamW optimizer, held-out validation, atomic `.emb` checkpointing, foraging, cloud teachers, and GitHub sync.
- **Native Inference Engine (`native/`)**: Written completely in **C++17** (`libember_infer.so`), compiled for `arm64-v8a` and `x86_64` via the Android NDK + CMake, and invoked directly from Kotlin over **JNI** (`NativeInference.kt`) with pre-allocated KV caching, ARM NEON SIMD dot products, context-window sliding, and lock-free atomic cancellation.

See [`docs/v3-architecture.md`](docs/v3-architecture.md) for the complete `.emb` binary format, tensor layout, tokenizer contract, and memory budgets.

---

## Build & Install Android APK (GitHub Actions)

1. Open **Actions → Build APK → Run workflow** on GitHub (or push to `main` / working branch).
2. Download the **Ember-debug** artifact containing the self-contained `Ember-debug.apk` (includes compiled `libember_infer.so`, embedded Python 3.13 runtime, `numpy`, `zstandard`, and the `ember` backend — no PC, external server, Termux, or API required).
3. Install on your Android 7.0+ (`minSdk 24`) device.

---

## Run Desktop Companion & Automated Tests Locally

```bash
# 1. Build native C++ inference library & standalone native tests
cmake -S native -B native/build
cmake --build native/build -j
./native/build/test_native

# 2. Install Python dependencies and run full test suite (backend + 35 fixtures + C++ logit/greedy interop)
pip install -r requirements.txt
pytest -v tests/

# 3. Start local desktop HTTP + RPC server on port 8000
./run.sh
```
