# EMB4: Llama-style training → GGUF → local llama.cpp

## Architecture and verified sources

The model is now `ember-llama`, checkpoint magic **EMB4**, format 4. Old EMB3
GPT checkpoints are incompatible and left untouched. This is **not** Meta's
pretrained Llama 3, its vocabulary, or an 8B model.

Reference inspected: https://github.com/meta-llama/llama3/blob/main/llama/model.py
(`ModelArgs`, `RMSNorm`, `apply_rotary_emb`, `repeat_kv`, `Attention`,
`FeedForward`, `TransformerBlock`, `Transformer`). Source was retrieved through
GitHub's contents API. Original Llama 3 uses theta 500000, adjacent complex pairs,
RMSNorm, bias-free Q/K/V/O, repeated KV heads and SiLU(gate) * up.

Inference stays pinned to official **llama.cpp b11836**, commit
`f5027fb21307eb264aed483242603d68c87b0fe3`, SHA-256
`5fdae446417f5f6371b9cab097fa2404e1c0de632b00eb76213889b18153f64b`.
Inspected `src/models/llama.cpp` and `src/llama-model.cpp`: architecture `llama`
uses `LLAMA_ROPE_TYPE_NORM` (adjacent pairs), `LLM_NORM_RMS`, SwiGLU,
GQA-shaped Q/K/V, optional tied output. No RoPE permutation is needed because
Ember trains in the same adjacent-pair basis as ggml and Meta's complex reference.
Do not apply HuggingFace's split-half permutation to these checkpoints.

Configuration: layers, query heads, KV heads, width, FFN intermediate size (or
width multiplier), vocabulary, context, theta, epsilon and tied/untied output.
Constraints: width divisible by query heads; even head dimension; query heads
multiple of KV heads. Float32, no biases, dropout zero and **unscaled** RoPE are
currently supported; unsupported dtype/scaling/bias/dropout is rejected, not
silently ignored. This implements original Llama 3-style RoPE, not Llama 3.1
long-context frequency scaling. Default phone: 2 layers, width 64, 2 Q heads,
1 KV head, FFN 256, context 64, requested vocabulary 1500. Tiny test configs
are separate from larger presets. There is no claim these fit every phone.

## Training and data

NumPy remains the embedded training framework because it already ships in the
Chaquopy APK; PyTorch is not an Android runtime dependency. `ember/autodiff.py`
is a small reverse-mode tape implementing only the supported decoder training
operations, including analytic RMSNorm/RoPE/SiLU/softmax gradients and grouped KV
gradient reduction. Numerical finite-difference tests cover projections and
norms; all trainable components must receive finite nonzero gradients. This
choice preserves actual in-app training, rather than moving it to a PC-only demo.

Causal cross entropy uses explicit shifted targets from the document stream,
with `-100` for ignored positions. Upper-triangular and document masks prevent
future-token and cross-document leakage. AdamW, clipping, warmup/cosine schedule,
validation without updates, cancellation and checkpoint resumption remain.
Float32 only; no unverified mixed precision or gradient checkpointing.

The existing SQLite corpus, ingestion/extraction, Parquet reader, compressed
imports, duplicate-content handling and source documents are retained. Corpus
iteration uses paged SQL; no full corpus text copy is required. The checkout
contains small Parquet test fixtures, **not an available production FineWeb
shard collection**. No full FineWeb training is claimed or performed.

Splits are deterministic document-ID BLAKE2 hashes with a small-corpus holdout
fallback. Split algorithm, validation IDs and corpus revision are persisted in
SQLite `training:split`. Tokenizer fitting now excludes validation documents.
Tokenizer sample budget defaults to 2,000,000 characters and is explicit;
training/validation stream budgets default to 3,000,000/250,000 tokens. The
`vessel.cast` configuration accepts `tokenizerChars`, `maxTrainTokens`,
`maxValTokens`; limits persist in SQLite and are reported in status. Budgets
select a reproducible insertion-order subset, **not** the full FineWeb corpus.
Documents beyond the budget remain preserved. Streams report sampled documents,
characters, tokens and truncated coverage; small single oversized documents
may be clipped to the explicit token budget. Do not report these as whole-corpus
training. Packed sequences include BOS/EOS and document IDs. Large future training
can increase the explicit budgets, subject to available memory; the mobile
trainer has conservative parameter/activation admission checks.

## Tokenizer

Retained byte-level BPE schema 4: 256 raw UTF-8 byte IDs, PAD/BOS/EOS/DOC/UNK
256–260, merges from 261. Deterministic pair-frequency tie breaking, Unicode
GPT-2 pretokenization, reversible byte alphabet and rank-ordered merges are
shared by training/export/native inference. This is **not** Llama 3's tiktoken
vocabulary. GGUF `tokenizer.ggml.model=gpt2`, `pre=gpt-2` works independently
of `general.architecture=llama`. Tests compare exact native IDs for Unicode,
whitespace, contractions, punctuation and newline handling. Streaming uses an
incremental UTF-8 decoder. Larger vocabulary shortens sequences but grows the
embedding/output cost; the requested vocabulary shrinks to observed merge count.

## Export and native runtime

`python -m ember.gguf_export checkpoint.emb model.gguf --quantization q8_0 --validate-native`

The converter validates configuration, exact tensor set/shapes, finite weights
and tokenizer size. GGUF v3 metadata includes `llama.*` dimensions, KV heads,
RMS epsilon, RoPE base/dimensions and explicit no scaling. Embedding and output
rows stay [vocabulary,width]. `x @ W` Q/K/V/O and gate/up/down matrices transpose
to GGUF row-major [output,input]. Norm gains stay vectors. No learned positions,
LayerNorm biases, fused GPT QKV or GELU remains in the model. Output is omitted
only when tied. Q8_0 stores FP16 scale then 32 int8 values per block; only eligible
projection matrices quantize, norms/embedding stay F32. Quantization is lossy.
Publication is atomic and malformed checkpoints cannot overwrite valid exports.

Production Android path remains Compose → repository → `InferenceEngine` →
`LlamaCppInferenceEngine` → UTF-8 JNI → `libember_llama.so` → static llama.cpp/ggml.
Python/RPC uses that same C ABI, never NumPy sampling or a cloud fallback.
CPU-only build, `arm64-v8a` and `x86_64`, NDK 26.1.10909125, CMake 3.22.1,
Gradle 8.7/Java17. MIT notices are included in the repository and APK.

SAF import copies to app-private model library; selection persists across
restart. Voice controls loading, generation, streaming, Stop, unloading, context,
threads, temperature, top-k/top-p, repetition and export quantization. Jobs run
off the UI thread with serialized native ownership and structured errors.
The native library contains no download/server dependency. INTERNET remains only
for explicit optional foraging/teachers/GitHub features. No token/prompt upload,
API key or network is needed for local generation.

## Verification status

Host build succeeded; a real four-step trained EMB4 F32 model loaded and generated
with pinned llama.cpp. **95 Python tests passed** at the first complete migration
run, including native tokenization/logit parity, gradient checks, tiny controlled
overfit, no-future leakage, actual checkpoint conversion CLI, native repeated
requests/cancellation/streaming/cleanup, and socket-prohibited generation.
This is pipeline correctness testing, not pretrained language competence.

The Android workflow still builds both ABIs, runs JVM/Python/native tests,
verifies APK contents and uploads `Ember-debug.apk`. Final CI outcome is reported
in the implementation response, not assumed here. No physical Android device or
emulator is available: install/start, JNI execution on Android, interactive SAF,
UI responsiveness and airplane-mode operation on a phone remain unverified.
