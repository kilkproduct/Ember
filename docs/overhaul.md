# Ember 0.7 engineering overhaul

Baseline source: `412f8c1c6a26440d1d1043e429a534d7e1fc5751`.
Host benchmark environment and raw samples: `benchmarks/*host.json`.
`OPENBLAS_NUM_THREADS=1 python scripts/benchmark_host.py` is repeatable, but host
CPU timings are not phone latency or thermal measurements.

## Reference inspection

Pinned official llama.cpp remains b11836 / f5027fb21307eb264aed483242603d68c87b0fe3.
Its `src/models/qwen2.cpp`, `qwen3.cpp`, `llama-model.cpp`, `llama-vocab.cpp`,
`llama-chat.cpp` and `include/llama.h` already implement the required dense Qwen
architectures, so upgrading the pin would add risk without resolving Ember's
actual blockers. Qwen2.5 uses `qwen2`; Qwen3 uses `qwen3` with per-head Q/K RMS
normalization and sometimes head dimensions independent of hidden_width/heads.
MoE, VL, next-generation hybrid and embedding/reranker variants are not accepted.

GGUF specification reviewed through ggml's official `docs/gguf.md`; quantization
block-size/byte pairs are copied from pinned official `gguf/constants.py`.
No generic validator assumes Qwen IDs or caps imported vocabularies at 50,000.

Compose reference: Android's compose-samples Jetchat README/source and existing
Material3/lifecycle APIs: native navigation, saveable screen state, lazy content,
lifecycle-aware flows, shared functional typography, and separate everyday vs
advanced controls. No gradient/glass, marketing dashboard or animation dependency
was added. Device frame timing, TalkBack and touch testing need an Android device.

## Actual features and bottlenecks

Fire is causal NumPy Llama training, not inference. Training sample intervals
also export GGUF and generate through llama.cpp. Checkpoints and validation are
separate real costs. Weight-gradient GEMMs now flatten batch/token dimensions,
avoiding batched weight-gradient temporaries and reductions. Gradient accumulation
uses in-place sums without aliasing the reverse tape. Finite-difference, resume,
masking and convergence tests remain essential gates.

Vessel's estimate is architecture allocation arithmetic, NOT completion time.
Previously each edit waited 320 ms and entered Python's RPC path. It is now an
immediate bounded Kotlin calculation matching the training formula. Fire's old
single-step countdown is replaced by median throughput over a rolling window,
unknown during startup and a rough minute range afterward. Casting exposes actual
BPE merge progress and named stages; it does not promise a completion time.

GGUF validation previously rejected everything except Ember Llama + classic BPE,
supported only F32/F16/Q8_0, and read up to 64 MiB into Python for each inspection.
It now uses a bounded virtual file mapping, signature-keyed inspection cache,
architecture-specific dense Qwen tensor checks, native tokenizer ownership and
common upstream quantizations. Native loading remains authoritative.

## Models, storage and inference

SAF models with persistable permissions and seekable file descriptors are
registered as external files without a multi-GB copy. Descriptors live in the
application repository; URI registrations reopen after restart. Revoked/moved
files produce a visible recovery notice. The original file is never deleted.
Non-seekable/cloud providers fall back to a disclosed cancellable chunked copy,
with storage checks and measured copied bytes. External registration is metadata
validation, not native readiness; first generation performs native validation.
Copied-model import still performs native loading and unloads the Python context
so it does not coexist with Android's JNI context.

Imported GGUF chat templates go through upstream `llama_chat_apply_template`.
That API supports a finite built-in template family, NOT arbitrary Jinja execution.
ChatML (including Qwen-style user/assistant control tokens) is exercised. Qwen3
thinking/tool-call customizations are not claimed. Imported BOS/EOG behavior comes
from upstream vocabulary flags; model-native EOG tokens stop generation.
A raw user prompt is one user turn; this is not an unlimited multi-turn chat store.

Session context can be set through 2048 and beyond (8192 safety cap), bounded by
native training context. No RoPE values are rewritten. Prompt overflow uses the
existing oldest-token removal policy and returns a warning. CPU threads, logical
batch and physical microbatch are separate controls; GPU remains unavailable.

## Data and retrieval

JSONL/NDJSON and CSV/TSV, including gzip/bzip2/xz wrappers, stream bounded previews
and batches into SQLite. Resume replays records deterministically and content
hashes prevent duplicate rows. No invented document total: source-read progress
is separate from final parsing completion. Other formats retain the existing
bounded (512 MiB) container/Parquet path; huge archives must be split or converted
to streaming formats. These are real limits, not claims of fully streamed Parquet.

Foraging uses up to four independent source workers and at most two requests per
host. Responses have an 8 MiB cap, finite timeouts, bounded retries, interruptible
backoff and HTTP 429/5xx handling. Writes/budget checks serialize; corpus summaries
are throttled. Real public APIs supply Wikipedia, HN, Gutenberg, OpenLibrary,
Stack Exchange and explicit Reddit archive sources. A source's unavailability is
logged; no source content or success counts are synthesized. DNS/socket waits may
last until the request timeout after cancellation. Live public-web operation
cannot be tested here because outbound hosts are restricted.

## Packaging and measurement

Universal debug APK retains both ABIs. Additional arm64-v8a and x86_64 APKs avoid
shipping the other ABI without deleting functionality from the target device.
Release builds enable R8/resource shrinking with JNI/reflection keep rules; signing
stays external. CI builds debug and unsigned release, builds the exact baseline
in a separate temporary source directory and uploads actual package/native sizes.
No model weights or development toolchain is packaged as an asset.

No physical device/emulator is available. Startup, frames, rotation, SAF provider
behavior, low-memory kills, sustained throughput and airplane-mode use on Android
remain unverified; no host benchmark should be presented as those measurements.
