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

## Executable architecture map

```
Compose + lifecycle StateFlow (MainActivity/MainViewModel)
       |                    Android-owned SAF URI / seekable PFD
EmberRepository ----------- /proc/self/fd/N (same application process)
       |
EmberEngine (IO coroutines, explicit task gates, streamed events)
       |                                 |
Chaquopy -> ember.backend            LlamaInference -> JNI
       |                                 |
SQLite Corpus / ImportManager        libember_llama.so
Forager (bounded source workers)      official pinned llama.cpp + ggml CPU
NumPy Llama + reverse tape / AdamW     native GGUF tokenizer, template, sampler
       |
EMB4 atomic checkpoints + optimizer state -> F32 / Q8_0 GGUF export
```

The desktop HTTP/browser client remains useful for host debugging; Android does
not route production inference through that server or through a JavaScript
engine. Model weights stay in app-private files or the retained external URI.
Python and JNI share a process but use separate native handles: preparation and
copied-file validation explicitly unload the Python sample handle before JNI use.
Large Python RPCs run on IO dispatchers, not the Compose thread. Cancellation
signals do not wait on the native generation mutex. Native chunks are byte arrays
so UTF-8 decoding is not corrupted by JNI modified-UTF8 conversion.

### Compatibility matrix: what has actually been exercised

| Architecture / format | Evidence | Not established |
|---|---|---|
| Ember dense Llama, F32 and Q8_0 | Train/export/load, tokenizer/logit parity, native stream/seed/cancel/reload tests; F32 lifecycle under ASan/UBSan/LSan | Device throughput, large trained-model quality |
| Dense `qwen2` (Qwen2 / Qwen2.5 graph family), F32 and Q8_0 | Synthetic graph fixtures with distinct tokenizer metadata, ChatML, EOG IDs, 2048 context, stream/cancel/recovery | Official pretrained checkpoint and full real Qwen vocabulary |
| Dense `qwen3`, F32 and Q8_0 | Same synthetic tests plus per-head Q/K RMS gains and independent head dimensions (head width differs from hidden width / head count) | Thinking-mode/tool APIs, official checkpoint quality |
| Q4_K / Q5_K / Q6_K / IQ and newer block encodings | Byte/block size table compared against **every** entry in pinned upstream constants; loading delegated to upstream kernels | End-to-end real Qwen files of each quantization |
| Qwen MoE, VL, hybrid/Next, embedding/reranker, adapters, split shards | Rejected by Ember's dense architecture/tensor admission | No compatibility claim |
| Arbitrary Jinja templates | Explicit error if upstream built-in formatter does not recognize the template | No general Jinja execution |

The tests named Qwen2/Qwen2.5/Qwen3 generate **small random fixtures**. Those names
must not be mistaken for downloaded official pretrained weights. Allowed network
hosts in this environment do not include model-hosting providers, and no real
pretrained Qwen file was supplied. Model licensing and trustworthy provenance
remain the model owner's responsibility; Ember neither downloads model weights
nor falls back to a cloud model.

Reference links:
- [Pinned upstream Qwen2 graph](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/src/models/qwen2.cpp)
- [Pinned upstream Qwen3 graph](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/src/models/qwen3.cpp)
- [Pinned native API / chat template contract](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/include/llama.h)
- [Pinned quantization constants](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/gguf-py/gguf/constants.py)
- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Jetchat reference inspected at 25631e8](https://github.com/android/compose-samples/tree/25631e8f544a9a1e3f01e296eaf1b4c28428196c/Jetchat)

The source archive checksum is
`5fdae446417f5f6371b9cab097fa2404e1c0de632b00eb76213889b18153f64b`.
`scripts/fetch-llama-cpp.sh` checks it before use; no floating llama.cpp download.

## Configurations, real settings and allocation policy

Own training uses float32, tied embeddings, RMSNorm, adjacent-pair RoPE and
SwiGLU. Dropout/bias/unsupported positional variants are rejected, not silently
accepted. Architecture controls include layer/head/KV-head counts, hidden width,
context, requested vocabulary, FFN multiplier or exact intermediate width,
RoPE theta, bounded tokenizer/corpus token budgets and initialization seed.
Custom named architecture configurations retain seed and preparation budgets.

Fire exposes batch/steps/LR, gradient clipping (0 disables), accumulation,
AdamW weight decay/betas/epsilon, automatic or explicit warmup, final LR fraction,
sampling/validation/checkpoint intervals, batch sampling seed, sampler/boundary/
balance/lesson-loss policies, sample prompt and saved-schedule/cursor resume.
Stopping during an accumulation cycle discards the incomplete optimizer update.
Non-finite loss/gradients fail the run without publishing those values as metrics.
Final checkpoints still run when periodic checkpoint interval is zero.

Own vocab maximum is **50,000**, not a promised result. BPE stops if there are no
remaining repeated pairs. It never pads a small corpus to a requested vocabulary.
The actual 50k regression uses a constructed valid merge table, not a claim that
a tiny corpus learned 49,739 merges. It tests embedding/output dimensions, stable
loss/gradients, checkpoint restoration, GGUF export and native tokenizer parity.
Imported GGUF vocabularies are never resized to this training limit.

For tied embeddings, let `L=layers`, `C=width`, `H=heads`, `K=KV heads`,
`F=FFN width`, `V=vocabulary`, `T=context`, `B=batch`. Parameter count is
`P = V*C + C + L*(2*C*C + 2*C*(C/H*K) + 3*C*F + 2*C)`.
The conservative training working-set estimate is
`16*P + B*L*(T*(32*C + 12*F) + H*T*T*16) + B*T*V*16` bytes.
This is an admission heuristic, not a measured peak or total Android process RAM;
Python/runtime/OS, allocator fragmentation and optimizer temporaries still matter.
The quadratic attention and vocabulary logits are included; FFN-rich presets no
longer pretend their activations cost the same as a narrow FFN.

Python refuses training working sets above 512 MiB and warns above 160 MiB.
Android additionally checks current system memory before cast/train and refuses
estimates exceeding 60% of currently available memory or a low-memory signal.
Native inference checks GGUF weight bytes + F32 KV + context working-set reserve
against 80% of current `MemAvailable` before weight/context loading. Missing kernel
availability data falls back to real allocator errors; this is not an OOM guarantee.
Threads are bounded 0–8 (0 = conservative auto, capped at 4); logical batch <=2048,
physical microbatch <= logical batch, CPU only, context <=8192 and native trained
context. GPU requests fail explicitly. Invalid UI values are rejected visibly.

Android severe thermal status (API 29+) and critical memory callbacks request
safe training/foraging stop and native generation cancellation. They do not stop
work merely because the UI went into the background. These callbacks and memory
admission are implemented but **not exercised on an Android device**; a hard LMK
can still kill the process before a final save. Last committed snapshots survive.

### Presets (requested-vocab estimates, batch 2)

| Preset | Layers / width | Heads / KV | Context | Requested vocab | Parameters | Working MiB |
|---|---:|---:|---:|---:|---:|---:|
| phone | 2 / 64 | 2 / 1 | 64 | 1,500 | 219,200 | 8.0 |
| tiny | 2 / 64 | 2 / 1 | 64 | 2,000 | 251,200 | 9.5 |
| small | 4 / 128 | 4 / 1 | 128 | 3,000 | 1,335,424 | 50.1 |
| medium | 6 / 192 | 6 / 1 | 192 | 4,000 | 3,940,800 | 157.8 |
| micro | 1 / 32 | 2 / 1 | 32 | 500 | 31,456 | 1.2 |
| deep-narrow | 8 / 64 | 4 / 2 | 128 | 2,000 | 620,608 | 43.3 |
| wide-shallow | 2 / 256 | 8 / 2 | 128 | 8,000 | 3,949,824 | 109.5 |
| gqa-4 | 4 / 128 | 8 / 4 | 256 | 4,000 | 1,496,192 | 138.1 |
| ffn-rich | 3 / 128 | 4 / 2 | 128 | 4,000 | 1,840,000 | 61.7 |
| long-context | 1 / 32 | 1 / 1 | 2048 | 2,000 | 80,480 | 264.2 |
| vocab-50k | 2 / 64 | 4 / 1 | 128 | 50,000 | 3,319,104 | 252.5 |

Legacy CLI `tablet` and `desktop` names remain; desktop's ~650 MiB estimate is
blocked by the phone policy. Presets are valid **shapes**, not tested phone-fit,
quality or speed recommendations. Actual corpus-limited vocabularies can reduce
these estimates. Long-context training is still quadratic attention, not Flash
Attention or a claim of practical phone throughput at 2048.

## Data integrity, resume and limitations

Explicit record `split` / `partition` (or `meta.split`) is retained. `dev`/`val`
map to validation; test/heldout records are excluded from both training and
model-selection validation, and all held-out IDs are excluded from BPE fitting.
Conflicting duplicate content adopts the **most protected** split (`test` >
`validation` > `train`) instead of silently discarding a later split label.
Without explicit labels, content-grouped stable document hashes drive holdout.
There is no destructive SQLite schema migration for this change.
If a new held-out label is applied to text already used to train existing weights,
recast before claiming independent evaluation: software cannot retroactively
untrain an existing checkpoint. Existing datasets imported by an older version
without saved labels cannot have those labels guessed back; reimport the source
with its explicit labels to protect future runs.

Streaming preview reads at most 256 KiB. Source-read bytes are not parsed-record
completion; buffered reads can reach 99% before parsing finishes. Compressed
streams expose decompressed bytes, not an invented compressed completion percent.
Persistent cursors count committed valid records; resume replays the source from
the start, rather than claiming O(1) random-access resume through gzip. Batches are
100 documents. A single line above 8 Mi characters fails a clear safety guard;
CSV's stdlib per-field bound also applies. Other container formats retain the
512 MiB parser bound, decompression ratio/entry/depth/traversal checks. Source
records longer than Corpus's existing 120,000-character document bound are still
truncated by that pre-existing policy; retain the source for archival fidelity.
No claim of unbounded full-fidelity single-document import is made.

Forage public-response cache: 1-hour TTL, <=64 entries, <=16 MiB total and <=8 MiB
per response; atomic disposable files, no deletion of corpus/model files. Requests
reserve per-host starts at least 0.5 seconds apart, with up to two admitted slots
per host including retry/backoff and four source workers overall. Resume saves
the current topic **before** fetching; interrupted topics replay and content hashes
avoid duplicate rows. Source failures and cache hits are separate real counters.
No live Reddit/Wikipedia/etc result or availability is claimed from mocked tests.

Imported same-named model files get a unique library suffix, never overwrite a
previous model. Seekable external registrations are capped at 32 open descriptors;
Forget closes the descriptor/releases the permission and removes only registration.
Fallback copying is cancellable and storage checked. Android-owned staged files
on the same internal volume are hard-linked into the model library after native
validation, avoiding a second multi-GB copy; ordinary host imports preserve the
original path. Native/copy cancellation is bounded by active provider/OS I/O;
metadata reads already blocked in a provider cannot be forcibly interrupted here.
SAF revocation, the final registration/cancel race and real multi-GB provider
behavior still require device testing. Do not equate structural registration with
native readiness, low-memory fit, a provider availability guarantee, or model quality.

Recast/import backs up trained weights before replacement and stops if that
backup fails. Incompatible legacy checkpoint backups are SHA-256 verified before
publication. Checkpoints publish optimizer state under a unique filename and
atomically replace the EMB4 reference; old committed data survives a failed write.

## UI / dependency and package decisions

Chat is the first destination. Its order is prompt, streamed response, expandable
sampling/resources, model library, then manual teaching. Data contains sources,
forage work and dataset lifecycle. Vessel is shape/preset/preparation/cast; Fire
is the actual training run and checkpoint workflow. A persistent active-work strip
keeps tasks visible when navigating. Removing tab `AnimatedContent` prevents two
whole work screens being composed during every transition. Themes follow system
light/dark, surfaces are compact squared Material3, and values/labels remain
visible rather than hidden behind icon-only dashboard cards.

No UI animation package was added: the existing Compose animation dependency is
still used by useful teacher/notice expanders. Core inference has no network/API
key dependency, analytics or tracking library. Existing optional user-enabled
teacher/GitHub features are retained and are separate network features, not a
core-inference fallback. Runtime APK Python dependencies are pinned to available
Chaquopy wheels `numpy==1.26.2`, `zstandard==0.23.0`. Host `gguf` and `pytest` are
conversion/test dependencies, not APK runtime additions. The pinned native backend
has TLS/HTTP/GPU/example/tool builds disabled. Static upstream ELF symbols are
hidden (`--exclude-libs,ALL`), retaining only Ember's public ABI/JNI and enabling
linker dead-section removal without modifying upstream architecture graphs.

R8/resource shrinking is enabled for release with explicit JNI/callback keep
rules; Compose tooling and UI-test manifest are debug-only. Universal debug APK
retains both architectures; phone-specific arm64 APK omits the x86_64 binary, not
phone functionality. APK CI's size JSON reports actual file bytes, each top-level
`.so` entry's raw/compressed sizes and all asset-container bytes separately. Native
Python extensions inside Chaquopy containers belong to the asset total, not a
misleading claim that the top-level `.so` sum includes everything extracted later.

## Measured host comparison

Same sandbox host, x86_64 Linux 6.1.158, Python 3.11.2, NumPy 2.4.6, 2 logical CPUs,
`OPENBLAS_NUM_THREADS=1`. Filesystem/model pages are warm; these tiny random models
are not representative of real multi-billion-parameter Qwen or Android. Baseline
Python samples were captured at `412f8c1`; the native baseline file was captured
after early validator edits but before native changes, so it is used only for
native load/generation comparison, not as a clean validator baseline. Final host
source: `ededf5e46d345bbc4c40a1eead31d7254b95e1f2` (clean worktree when sampled).
Subsequent code changes are Android-only. Full samples/source hashes are retained
in `benchmarks/after-host.json` and the paired baseline files.

| Fixed benchmark | Baseline median | After median | Interpretation |
|---|---:|---:|---|
| 2-layer/128-width training, batch 2/context 32 | 17.570 ms/step | 8.826 ms/step | ~49.8% lower observed wall time; GEMM/tape allocation optimization |
| Tokenizer, 200 documents/requested vocab 500 | 195.239 ms | 128.221 ms | Observed samples; no claim of a new BPE algorithm or statistically proven improvement |
| Tiny checkpoint save | 13.595 ms | 9.219 ms | Filesystem/allocator variance; publication's full-weight reload removal is separate |
| Tiny checkpoint load | 1.572 ms | 1.586 ms | Essentially unchanged |
| Repeated GGUF inspection | 2.071 ms | 0.00835 ms | New signature-keyed warm-cache path; not first model load or disk-cold timing |
| Native tiny model load | 3.204 ms | 3.312 ms | No speedup claim; includes new metadata/memory admission |
| Native 32-token generation | 9.431 ms | 9.925 ms | No improvement claim; ~5% slower in these short samples |
| Isolated CE, `[2,128,50000]` | 104.275 ms | 26.103 ms | Reused stable softmax/gradient buffer instead of full log-probability/probability temporaries |
| CE isolated process peak RSS | 336,196 KiB | 192,508 KiB | ~42.7% lower observed peak; not total Android training memory |

CE loss is `11.202529907226562` in both runs; float64 gradient/loss and non-mutation
oracles cover the rewrite. The CE benchmark loads the old function through
`git show` in a separate process, without changing branches or touching user data.
Tiny native after-only first-token callback median: **0.542 ms**; callback Stop to
return median: **0.320 ms**. No matching baseline was captured for those metrics,
so they are not presented as before/after improvements. These are host callback
measurements, not UI-visible phone latency or cancellation of stalled provider I/O.

Reproduce from repository root with a Python environment containing requirements:

```sh
scripts/fetch-llama-cpp.sh
cmake -S native/llama -B native/build-llama -DCMAKE_BUILD_TYPE=Release -DEMBER_BUILD_LLAMA_TESTS=ON
cmake --build native/build-llama -j2
export EMBER_LLAMA_LIB="$PWD/native/build-llama/libember_llama.so"
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/
OPENBLAS_NUM_THREADS=1 python scripts/benchmark_host.py
OPENBLAS_NUM_THREADS=1 python scripts/benchmark_loss_memory.py --baseline-ref 412f8c1c6a26440d1d1043e429a534d7e1fc5751
OPENBLAS_NUM_THREADS=1 python scripts/benchmark_loss_memory.py
OPENBLAS_NUM_THREADS=1 scripts/test-native-sanitizers.sh
```

Host result: **139 passed**, two expected overflow warnings from the intentionally
absurd-learning-rate refusal test. Native regular lifecycle tests cover F32/Q8_0;
ASan/UBSan/leak checks cover the F32 native harness. The sanitizer CI workflow at
[f6c6ef2 / run 40011387527](https://github.com/kilkproduct/Ember/actions/runs/40011387527)
passed. JVM contracts cover expanded configuration round-trip, allocation/presets,
ETA/counters and compressed unknown-total handling. Android instrumentation smoke
is compiled by APK CI; actual execution still needs a device/emulator.

## Build blockers found (not suppressed)

- First Compose compile exposed a missing `EmberTextField` import in Vessel.
- The newly enabled release build exposed a platform theme inheritance cycle
  (`Theme.Ember.Base` vs `Theme.Ember`). The base was renamed to a dotless style,
  retaining the real framework parent; lint was not disabled or baselined away.
- Notification posting now checks runtime permission at the actual call site and
  handles a concurrent revocation. A user refusing notifications does not disable
  core local inference.
- Native quantization table review caught Q8_1's pinned 36-byte (not 40-byte)
  blocks and added the upstream Q2_0 entry; the full pinned table is now a test.
- GGUF malformed/truncated header errors close mappings in `finally`, duplicate
  metadata/tensors and unaligned/overlapping tensor payloads fail explicitly.
- Dataset JSONL detection no longer treats a malformed first record as permission
  to turn a whole later JSONL file into one plain-text document.
- External file-descriptor selections keep `/proc/self/fd/N` identity rather than
  resolving it to a provider-private path, and stale persisted numeric FDs are not
  trusted before Android reopens their URI grants after process restart.
- Explicit dataset labels were previously discarded; they now survive import,
  deduplication and stream rebuilding. Same-named copied models no longer replace
  previous user files.

## Install / first run

Download the `Ember-debug` artifact from the final verified APK workflow run. It
contains universal `Ember-debug.apk`, phone-specific `Ember-arm64-v8a-debug.apk`,
`Ember-x86_64-debug.apk`, and before/after size JSONs. Prefer arm64 on a matching
phone. These are debug-signed installable APKs; release is also compiled with
shrinking but needs the owner's signing key before distribution.

1. Update with the same signer; preserve data using the backup instructions.
2. Chat → model library → Import GGUF → choose a **local, dense** supported model.
   Check external URI vs app copy, then select it and generate. Registration alone
   is not a native load result. Start with context 2048 when supported, threads 0,
   batch 256/microbatch 64; reduce context/resources if memory admission refuses.
3. Expand Generation settings for token cap, seed, sampler and CPU controls.
   Native context/resource values appear after load. Oldest-token removal is
   explicitly warned; no unlimited-context claim is made.
4. To train instead: Data → add/import actual text → Vessel → choose shape or
   load a named configuration → Cast → Fire. Casting fits BPE/initializes weights;
   it does not learn next-token prediction. Fire is the real optimizer loop.
5. Use pause/resume for datasets and saved-topic resume for foraging; Douse/Stop
   are the real cancellation paths. See [device verification](device-verification.md)
   for restart, corruption/revocation, offline and lifecycle checks.

The final APK source/CI link, package byte counts and PR are recorded alongside
this report after the build completes. This document's unverified-device limits
still apply even when compilation and package verification pass.
