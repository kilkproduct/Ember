# GGUF and Parquet repair (2026-10-23)

## Baseline and root causes

Baseline: `3ef376f31c56c646aba5070107077677ed7450cf`, branch `arena/6dba3424-ember`.

The exact message **“This container exceeds the bounded legacy parser limit”** comes from `ImportManager.stage_path` in `ember/ingest.py`, not Arena or a third-party Parquet library. Only JSONL/CSV/TSV had file-streaming paths. Everything else, including Parquet, went through `Path.read_bytes()` and a 512 MiB legacy-container guard. The old custom Parquet decoder additionally accumulated all selected column values for an entire row group and silently substituted empty strings for unsupported encodings. DATA_PAGE_V2 was incorrectly decompressed before separating its uncompressed definition levels. Corpus storage silently truncated texts at 120,000 characters.

A reproducible GGUF rejection was found in `read_gguf_header`: its mmap covered only the first 64 MiB. Valid metadata extending beyond that prefix raised `truncated GGUF string`. The regression inserts an aligned, valid ignored metadata string into a tiny model and verifies actual llama.cpp generation. Restoring the old mapping reproduces the failure; see `verification/gguf-prefix-baseline.txt`. This establishes a real defect, **not that this was the cause of an unspecified user's model failure**. There was no supplied model, device stack trace or logcat. Small valid Llama and synthetic Qwen graphs already loaded on the host baseline.

External SAF registrations previously did structural inspection only, unlike copied imports. They now require native model/tokenizer/context validation before registration. Native loader errors now include upstream error-level diagnostics rather than discarding the detailed reason. Android releases its JNI model before Python native import validation to avoid simultaneous resident copies.

## Backend and runtime selection

Inference remains local CPU llama.cpp, unchanged pin: tag `b11836`, commit `f5027fb21307eb264aed483242603d68c87b0fe3`. Source tarball SHA-256 is verified by `scripts/fetch-llama-cpp.sh`. Android statically links llama and ggml CPU into `libember_llama.so` for arm64-v8a and x86_64. No cloud fallback.

Parquet uses the existing **in-tree Python reader**, refactored to file-backed mmap, projected flat string columns and independent page iterators. There is no new mobile Parquet dependency/version. PyArrow/pandas are not Android runtime dependencies; desktop wheels are not usable as Chaquopy Android binaries. Adding the Hadoop/JVM Parquet stack or Arrow native runtime without target-ABI validation would not be a verified replacement. This repair deliberately retains and tests the limited mobile reader instead of claiming a fully general maintained Parquet implementation. `pyarrow==23.0.1` is an independent **host fixture writer only**, pinned in the dev extra and APK workflow. Mobile zstandard remains `0.23.0` in Gradle.

Supported/tested Parquet subset: flat UTF-8 BYTE_ARRAY columns, optional/required values, PLAIN and dictionary encodings, V1 and V2 pages, uncompressed/Snappy/Gzip/Zstd. Nested/repeated schemas, unsupported codecs/encodings, invalid indices and truncated values fail explicitly. Numeric/binary logical columns are not extracted. Required content must be a recognized string column (`text`, `body`, `content`, etc.); URLs are not silently treated as content if a text cell is null.

## Pipeline and integrity

- Footer-only schema preview bypasses both `read_bytes` and legacy containers.
- Independent column generators keep a dictionary and decoded page per selected string column, never full row groups. mmap reserves file address space, not a whole Python byte array. Memory still depends on encoded page/dictionary sizes and individual document sizes; this is **not** a hard RSS cap against hostile huge pages.
- Transactions flush at 100 rows or about 1 Mi-character of text (one oversized record is preserved, not truncated). Accepted text and string provenance fields (`id`, URL, date, language, dump, etc.), shard name and row index are persisted in SQLite. Existing content hashes and split protection remain in force.
- Resume checkpoints advance only after successful batch persistence; a crash between the SQLite batch and manifest update replays at most a batch, using existing deduplication. Pause flushes pending work; cancellation discards the uncommitted batch. Failed transactions retain prior commits and the prior cursor.
- File identity (device/inode/size/mtime_ns) is persisted to reject changed staged files on resume. This is **not cryptographic verification** of completed shards. Files are not automatically skipped as completed based on names.
- Real counters include committed rows, inserted/merged documents, rejected null/blank texts, row groups decoded, and projected page/footer bytes. Replayed bytes are explicitly labeled. Percent uses committed rows against footer rows, not source byte offsets. Elapsed time and session-average rows/second are reported; no speculative ETA is shown. UI update events are throttled to 150 ms.
- SAF dataset staging uses streaming I/O into durable app-private `filesDir/imports` for restart/resume, with low-storage checks and coroutine cancellation. Arbitrary 8/16 GiB staging caps are removed; available disk storage remains a real constraint. Model SAF descriptors are retained without copying where seekable/persistable access is available; non-seekable providers use streaming copies.

## Verification performed locally

Host: Linux, Python 3.11, NumPy 2.4.6, PyArrow 23.0.1. Android CI uses Python 3.13 and its established Gradle/NDK workflow.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pyarrow==23.0.1 cmake ninja
PATH="$PWD/.venv/bin:$PATH" scripts/build-native.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q tests
.venv/bin/python scripts/make_test_gguf.py /tmp/f32.gguf --steps 4
.venv/bin/python scripts/make_test_gguf.py /tmp/q8.gguf --quantize q8_0 --steps 4
printf 'EMB3 legacy incompatible checkpoint' > /tmp/not-gguf.emb
native/build-llama/test_llama_engine /tmp/f32.gguf /tmp/not-gguf.emb
native/build-llama/test_llama_engine /tmp/q8.gguf /tmp/not-gguf.emb
```

Observed: **162 Python tests passed**, two expected overflow warnings from the intentional diverging-training test. Both native fixture runs passed. Logs in `docs/verification/`. Regression coverage includes actual native inference through `/proc/self/fd` with a nonzero original descriptor offset, repeated load/unload, corrupt tensors and recovery, large metadata, genuine Parquet round-trips, optional/null/Unicode text, multi-page/multi-group processing, duplicate shards, pause/resume after manager recreation, cancellation, incremental persistence and simulated disk-full failure. The large-group test asserts <4 MiB traced Python allocations for 12,000 roughly 2 KiB texts in one row group; it is not an Android RSS measurement.

Verified architectures: tiny generated Llama F32/Q8_0 and synthetic dense `qwen2` (including a Qwen2.5-labelled graph) and `qwen3`, F32/Q8_0, with tokenizer, chat formatting, context, generation, streaming/cancellation. Synthetic fixtures have random locally generated weights; they are not pretrained Qwen releases and provide no quality evidence. No real downloaded Qwen model was used, so there is no upstream model license/revision claim. Qwen3-MoE/Next and multimodal variants remain outside Ember's structural whitelist.

Verified dataset schemas: synthetic FineWeb-like `{text,url,id,language,dump}` string columns and text-only shards, null values, and existing checked-in fixture schemas. No real FineWeb shard was supplied or downloaded. No claim of all FineWeb releases/codecs/nested schemas.

## Android build and remaining verification

The sandbox has no Java, Gradle, Android SDK, NDK, emulator or device. Use the repository's **Build APK** workflow on the implementation commit. It runs Python/native/JVM tests, assembles ABI-specific/universal APKs and checks packaged native libraries and Python modules. Its run/artifact URL and observed outcome must be reported separately; a baseline APK is not evidence for this implementation.

Unverified: physical-device inference, actual Android provider permission revocation/background lifecycle, Android memory pressure and very large real-world shards. The importer exposes one file at a time in the current UI; multiple independent shards can be imported sequentially, but no new multi-selection queue was added. Resume replays rows from the start rather than seeking directly to a row-group checkpoint. Numeric provenance is not retained. Throughput is session-average, not a rolling ETA. Foreground-service behavior is inherited, not device-validated. The limited in-tree decoder is not a general replacement for Apache Parquet/Arrow.

## User workflow

1. In Voice/model library, choose Import GGUF and select a supported dense model with Android's document picker. Retain read permission. Seekable documents remain external; other providers require space for a private streaming copy. Native validation must succeed before registration. Imported does not mean a JNI generation context is resident: generation loads it with the requested settings. Errors retain upstream loader details.
2. In dataset import, select `.parquet` directly (do not rename it). Wait for streaming staging and footer/schema preview, then confirm. Committed document/row counters, row groups and projected bytes update as processing proceeds. Pause retains the staged file; resume uses its saved cursor. Disk-full/corruption errors preserve completed batches. Import other shards sequentially; the report applies only to each selected shard.
