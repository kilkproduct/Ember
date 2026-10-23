# Ember v3 Architecture & Binary Specification

Ember v3 replaces the legacy WebView + TensorFlow.js engine with:

1. **Frontend UI**: Native Kotlin + Jetpack Compose on Android (`android/app/src/main/java/com/kilkproduct/ember/ui/`), plus a thin browser presentation client (`web/app.js`) talking over HTTP JSON-RPC to `ember.server` on desktop.
2. **Python Backend (`ember/`)**: Corpus persistence (SQLite WAL), multi-format streaming dataset ingestion, byte-level BPE training, token stream batching and masking, NumPy GPT forward & backward passes, AdamW optimizer, held-out validation, checkpointing, cloud teachers, public-web foraging, and GitHub sync. Packaged directly inside the Android APK via **Chaquopy 17.0** (CPython 3.13 + prebuilt Android `numpy` & `zstandard` wheels).
3. **Native C++17 Inference Engine (`native/`)**: Compiled via Android NDK + CMake into `libember_infer.so` (`arm64-v8a` and `x86_64`) and called directly over JNI from Kotlin (`NativeInference.kt` -> `native/src/jni_bridge.cpp` -> `ember::Engine`), as well as via `ctypes` (`NativeInferenceBridge`) for periodic sample generation and deterministic parity tests.

---

## 1. Shared Byte-Level BPE Tokenizer Contract (`version = 3`)

- **Base byte tokens (`0..255`)**: Single raw UTF-8 byte `b = id`. Any arbitrary UTF-8 string (including CJK, RTL scripts, and multi-byte emojis) encodes with zero `<unk>` tokens.
- **Special tokens (`256..260`)**:
  | Name | ID | Role |
  | --- | --- | --- |
  | `<pad>` | `256` | Batch right-padding token |
  | `<bos>` | `257` | Document / prompt start token |
  | `<eos>` | `258` | Document / response end token |
  | `<doc>` | `259` | Reserved structural separator |
  | `<unk>` | `260` | Reserved fallback ID |
- **Learned merges (`261 .. vocab_size - 1`)**: Ordered `[left, right]` pairs serialized as 1-codepoint-per-byte strings (`ensure_ascii=True`).
- **Pre-tokenizer**: Identical byte-class splitter in `ember/spec.py` (`pretokenize`) and `native/src/bpe.cpp` (`ember::pretokenize_bytes`). Whitespace runs attach a single leading space to the immediately following word or punctuation run.

---

## 2. Shared GPT Architecture (`arch = "ember-gpt"`, `format = 3`)

Pre-LayerNorm decoder-only Transformer with learned absolute positional embeddings and tied input/output embeddings:

```
x = wte[token_id] + wpe[pos]                      # [B, T, C]
for l in 0 .. n_layer - 1:
    h1    = LayerNorm(x, ln1.gain, ln1.bias, eps=1e-5)
    q,k,v = split_3(h1 @ attn.qkv.weight + attn.qkv.bias)
    att   = softmax((q @ k^T) / sqrt(head_dim) + causal_or_doc_mask)
    x_mid = x + ((att @ v) @ attn.proj.weight + attn.proj.bias)
    h2    = LayerNorm(x_mid, ln2.gain, ln2.bias, eps=1e-5)
    act   = gelu_tanh(h2 @ mlp.fc.weight + mlp.fc.bias)
    x     = x_mid + (act @ mlp.proj.weight + mlp.proj.bias)
xf     = LayerNorm(x, ln_f.gain, ln_f.bias, eps=1e-5)
logits = xf @ wte^T                               # [B, T, V]
```

- **Activation (`gelu_tanh`)**: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
- **Precision**: Little-endian IEEE-754 `float32` throughout. Verified Python vs C++ logit agreement is `< 1e-6` max absolute difference across full sequences.
- **Mobile KV Cache**: `ember::GPTModel` pre-allocates `[n_layer, block_size, n_embd]` key/value buffers once on load, prefilling prompts in `O(P)` and decoding subsequent tokens in `O(t)` with zero per-token heap churn and ARM NEON `vfmaq_f32` dot products.

### Exact Tensor Order & Shapes (`ember.spec.tensor_specs`)

| Tensor Name | Shape |
| --- | --- |
| `wte` | `[vocab_size, n_embd]` |
| `wpe` | `[block_size, n_embd]` |
| `blk.{i}.ln1.gain` | `[n_embd]` |
| `blk.{i}.ln1.bias` | `[n_embd]` |
| `blk.{i}.attn.qkv.weight` | `[n_embd, 3 * n_embd]` |
| `blk.{i}.attn.qkv.bias` | `[3 * n_embd]` |
| `blk.{i}.attn.proj.weight` | `[n_embd, n_embd]` |
| `blk.{i}.attn.proj.bias` | `[n_embd]` |
| `blk.{i}.ln2.gain` | `[n_embd]` |
| `blk.{i}.ln2.bias` | `[n_embd]` |
| `blk.{i}.mlp.fc.weight` | `[n_embd, ffn_dim]` |
| `blk.{i}.mlp.fc.bias` | `[ffn_dim]` |
| `blk.{i}.mlp.proj.weight` | `[ffn_dim, n_embd]` |
| `blk.{i}.mlp.proj.bias` | `[n_embd]` |
| `ln_f.gain` | `[n_embd]` |
| `ln_f.bias` | `[n_embd]` |

---

## 3. `.emb` Binary Container Format

```
Bytes 0..3   : ASCII magic "EMB3"
Bytes 4..7   : uint32 LE version = 3
Bytes 8..11  : uint32 LE header_len
Bytes 12..   : UTF-8 JSON header (ModelConfig + metadata)
Next 4 bytes : uint32 LE tensor_count
Repeated tensor_count times:
  uint32 LE name_len | UTF-8 name | uint32 LE dtype (0 = f32) | uint32 LE ndim
  ndim * uint32 LE dims | zero-padding to 16-byte alignment | raw LE float32 data
Final 4 bytes: IEEE 802.3 CRC32 over all preceding bytes
```

Saves write to `<path>.tmp`, `fsync`, and atomically rename via `os.replace`.

---

## 4. Resource Budgets by Vessel Preset

| Preset | Layers | Heads | Embd | Context | Target Vocab | Params | Approx. Train RAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `phone` (default) | 2 | 2 | 64 | 64 | 1,500 | ~0.20M | ~4 MB |
| `tiny` | 2 | 2 | 64 | 64 | 2,000 | ~0.23M | ~5 MB |
| `small` | 4 | 4 | 128 | 128 | 3,000 | ~1.19M | ~24 MB |
| `medium` | 6 | 6 | 192 | 192 | 4,000 | ~3.48M | ~72 MB |

Configurations exceeding 512 MB estimated training footprint are blocked prior to allocation to protect Android processes from low-memory kills.

---

## 5. Legacy JavaScript Brain Handling

Legacy `ember-brain` v1–v3 JSON exports from the removed TensorFlow.js engine are detected explicitly during file inspection and import:
- Weights from legacy JS brains are rejected with a clear message identifying the legacy format version rather than failing silently.
- Any `corpusDocs` embedded inside a legacy brain export are automatically recovered into the SQLite corpus so the user can cast and train a v3 model immediately without losing training material.

---

## 6. Cross-Boundary Invariants Added by the Debugging Pass

These invariants are enforced by tests, not conventions. Each one existed as a
silent failure mode before it was pinned down.

### 6.1 Event and payload contract (Python → Kotlin)

`EmberRepository.handleEvent` only reacts to these names, so the Python backend
must emit exactly them:

| Event | Payload contract | Consumer |
| --- | --- | --- |
| `train.progress` | `TrainingProgress` keys incl. `until`, `startStep` | header/Fire progress bar |
| `train.validation` | `step, loss, bestLoss, ppl, windows` | Heat chart markers |
| `train.sample` | `step, text, tokensPerSec` | Live samples |
| `train.autosave` | `name, step, bytes, mtime` | "Autosave written" notice |
| `train.finished` | `outcome, error, model, corpus` | completion/douse/failure notices |
| `model.castProgress` | full `ModelStatus` (incl. `castProgress`, `castStage`) | Vessel progress bar |
| `model.cast` | full `ModelStatus` | clears `casting` |
| `model.imported` | full `ModelStatus` | brain import |
| `corpus.changed` | `CorpusSummary` | Fuel counters |
| `forage.doc` / `forage.finished` | `title, documentsAccepted` / `+ error, corpus` | Fuel forage panel |
| `teachers.state` / `teachers.exhausted` | `teachers[], running` / `name, why` | Teachers tab |
| `import.progress` / `import.finished` | `ImportProgress` keys / `token, outcome` | Import sheet |

`train.done` is still emitted as an alias for non-Android consumers.
`tests/test_android_contract.py` asserts every payload key; the Kotlin mirror is
`android/app/src/test/.../EmberV3ContractTest.kt`.

### 6.2 Generation summary over the C ABI

`ember_engine_generate` returns only a token count, so the context-truncation
warning and the stop reason travel through
`ember_engine_last_generate_summary()` (JSON: `warning`, `stoppedBy`,
`promptTokens`, `newTokens`, `seed`, `elapsedMs`, `tokensPerSec`). The Python
ctypes bridge reads it, which is why `gen.run` can report "prompt exceeded the
context window" when generation ran in C++ instead of NumPy.

### 6.3 Deterministic identity and reproducibility

* Document ids are `blake2b(normalized text)[:12]`, not random UUIDs. The
  train/validation holdout is a hash of the document id, so random ids used to
  reshuffle the split on every import.
* `vessel.cast` accepts an optional `seed`; with a fixed seed and identical
  corpus text, casting and training are bit-reproducible
  (`test_cast_is_reproducible_with_an_explicit_seed`).
* Nested Android cast payloads (`{"config": {...}, "lr": ...}`) and flat
  desktop payloads are both accepted by `vessel.cast` / `vessel.estimate`.

### 6.4 Native memory and concurrency gates

`native/tests/test_native.cpp` runs clean under three toolchains in CI:

* `-O3` release,
* `-fsanitize=address,undefined -Wall -Wextra -Werror` with LeakSanitizer,
* `-fsanitize=thread` (the BPE encode cache is shared by the JNI thread and the
  Python bridge; it is mutex-guarded, and the concurrency test deliberately
  exercises the uncached write path).

Model loading additionally rejects integer-overflow tensor shapes, truncated
payloads, duplicate tensors, and non-finite (NaN/Inf) weights *before* the model
is marked ready, so corrupt files can never produce NaN logits on device.
Decoding sanitizes UTF-8 with maximal-subpart U+FFFD replacement to match Python
`bytes.decode("utf-8", errors="replace")`, including embedded NUL bytes.
