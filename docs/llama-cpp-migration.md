# llama.cpp migration and verification audit

## Selected architecture

Upstream llama.cpp **b11836** (`f5027fb21307eb264aed483242603d68c87b0fe3`),
source retrieved from the official ggml-org repository and SHA-256 checked
(`third_party/llama.cpp.version`). Resources inspected directly:
`docs/android.md`, `include/llama.h`, `src/models/gpt2.cpp`,
`conversion/gpt2.py`, `src/unicode.cpp`, `gguf-py/gguf/{constants,vocab,gguf_writer}.py`,
`ggml/src/ggml-common.h`, and `ggml/src/ggml-cpu/vec.h`.

GPT-2 is supported upstream and matches Ember's computation: learned positions,
pre-LayerNorm with bias, fused QKV and attention-output projections with biases,
GELU MLP with biases, final LayerNorm, tied output embeddings. No RoPE or GQA.
There is no justification for maintaining a custom llama.cpp architecture fork.
The maintained boundary is an explicit tensor layout + tokenizer + GGUF export.

### Tensor mapping (NumPy shape → ggml dimensions)

| Ember checkpoint | GGUF | Conversion |
|---|---|---|
| `wte` [V,C] | `token_embd.weight` {C,V} | unchanged; tied output omitted |
| `wpe` [T,C] | `position_embd.weight` {C,T} | unchanged |
| `ln_f.gain/bias` | `output_norm.weight/bias` {C} | unchanged |
| `blk.N.ln1.gain/bias` | `blk.N.attn_norm.weight/bias` | unchanged |
| `blk.N.attn.qkv.weight` [C,3C] | `blk.N.attn_qkv.weight` {C,3C} | transpose, contiguous |
| `blk.N.attn.proj.weight` [C,C] | `blk.N.attn_output.weight` {C,C} | transpose |
| `blk.N.ln2.gain/bias` | `blk.N.ffn_norm.weight/bias` | unchanged |
| `blk.N.mlp.fc.weight` [C,F] | `blk.N.ffn_up.weight` {C,F} | transpose |
| `blk.N.mlp.proj.weight` [F,C] | `blk.N.ffn_down.weight` {F,C} | transpose |
| projection biases | corresponding `.bias` tensors | unchanged |
| untied `lm_head.weight` [V,C] | `output.weight` {C,V} | unchanged |

GGUF uses reversed NumPy dimensions, `ne0` fastest. This is the same transpose
performed by upstream's GPT-2 Conv1D conversion. Header counts, typed metadata,
32-byte alignment and tensor offsets are independently tested using the official
`gguf` reader, and the output is loaded/evaluated by llama.cpp itself.

### Tokenizer and version boundary

`ember/gpt2_tokenizer.py` mirrors upstream's `unicode_regex_split_custom_gpt2`
scanner, including **literal space** prefixes, contractions, Unicode letters and
numbers, trailing whitespace runs, and upstream's 25-codepoint whitespace set.
The previous byte-class chunker is not tokenizer-compatible. Tokenizer schema
v4 rejects old v3 JSON, preserving its file and the corpus instead of silently
reinterpreting trained IDs. New casts use the compatible tokenizer.

IDs 0–255 remain UTF-8 bytes; 256–260 are pad/bos/eos/doc/unk. GGUF vocabulary
strings use GPT-2's byte-to-Unicode alphabet (`Ġ` for space); merges are rank-ordered
`"a b"` strings. Specials have CONTROL type and UINT32 IDs. BOS is prepended once
by generation, matching training, rather than implicitly added twice.

### Quantization and precision

F32 export retains every tensor. Q8_0 export quantizes only large block weight
matrices whose fastest dimension is divisible by 32; biases, embeddings and
positions stay F32. A Q8_0 block is **FP16 scale first, then 32 signed bytes**.
Unsupported quantization names fail clearly. Conversion validates all tensors
and finishes with fsync + atomic rename, retaining a previous valid destination
on failure. Export contains no timestamp so the same inputs produce identical bytes.

Upstream enables `GGML_GELU_FP16` even for its F32 CPU GELU kernel. This differs
from NumPy's analytic tanh evaluation: the deterministic untrained fixture's
maximum last-logit error was approximately **1.26e-4**. Tests use an explicit
**2e-4** absolute/relative bound, plus exact tokenizer and repeated greedy-token
checks. F32 KV is used for Ember's small models. These results are host CPU
measurements, not Android performance claims.

## Production paths and ownership

- Android: Compose → repository → `InferenceEngine` → `LlamaCppInferenceEngine`
  → `LlamaInference` JNI → `libember_llama.so` → pinned llama.cpp/ggml.
- Desktop/RPC/training samples: `Trainer.generate_sample` → `LlamaCppEngine`
  ctypes → the **same** C ABI/library. There is no NumPy fallback.
- Python remains embedded for training, SQLite, ingestion, GGUF export and optional
  network features. It is never invoked once per generated token.
- Prompt preparation is IO-confined; native loading and generation are off the UI
  thread. Kotlin streams decoded pieces through an app-scoped flow with request
  IDs. Stop sets an atomic native flag. A handle-lifetime monitor prevents
  cancellation racing destruction, without holding a compute lock.
- Each request clears llama.cpp memory and creates/frees its sampler chain.
  Upstream's `llama_sampler_sample` already accepts tokens: do not double-accept.
- Learned positions cannot be shifted as if they were RoPE. On context rollover,
  re-evaluate the newest context from position zero, never exceed the model's
  learned position table even when upstream pads allocated context to 256.
- Model library import copies through SAF/app-private files, validates metadata,
  dimensions and tensors, and requires native loading. Selection is stored in
  SQLite. Native unload releases RAM without deleting files.

## Files removed / retained

Removed: `native/src/*`, `native/include/*`, old `native/CMakeLists.txt`,
`NativeInference.kt`, `ember/sampler.py`, old custom-native tests and ctypes bridge.
The APK does not build `libember_infer.so`. NumPy training kernels remain as the
numerical reference. Training checkpoints and source documents remain.

Added: `native/llama` (C ABI + JNI + CMake), `ember/inference.py`,
`ember/{gpt2_tokenizer,gguf_export}.py`, Kotlin inference interface and implementation,
fetch/build/fixture scripts, source manifest and license, conversion/tokenizer/
llama.cpp tests. Updated Trainer, backend, Compose Voice/model controls, model
selection persistence, Gradle and APK workflow.

## Offline and resources

No HTTP client or server is linked into the native engine; all weights/tokenizer
metadata are local. Tests forbid Python socket connections while generating.
CPU-only is the reliable default. Defaults: context 512 (clamped to trained
context), prompt batch/microbatch 256, auto threads capped at 4, mmap on, tensor
checking on, generation capped at 8192. Voice exposes context and F32/Q8_0 export.
No automatic model download exists. Optional foraging/teachers/GitHub explain
INTERNET permission; none participates in ordinary generation.

## Verification checklist (actual, not inferred)

Host tests execute native initialization, invalid/missing files, GPT-2 GGUF loading,
upstream tokenization parity on contractions/Unicode/whitespace, last logits vs
NumPy, generation and EOG/cancellation paths, streaming, deterministic repetitions,
Q8_0 loading, unload/reload and model management. Native tests preserve assertions
in Release builds (`-UNDEBUG`). Conversion tests independently parse via gguf-py.
The full pinned native dependency and wrapper were built and exercised under
ASan + UBSan with LeakSanitizer for both trained F32 and Q8_0 fixtures.

The APK workflow fetches verified sources, builds the host engine, runs fixtures
and Python tests, runs the HTTP workflow, builds both Android ABIs, compiles JVM
contract tests, verifies APK contents and uploads the artifact. Final run results
are reported in the delivery message, not predicted here.

### Remaining limitations / checks requiring hardware

- No emulator or physical Android device is available here. Install/start, JNI
  callbacks on-device, airplane-mode interaction, SAF picker and OS process kills
  remain device checks, not claimed successes.
- CPU only. GPU-layer parameters are reserved for future backends; no working
  Vulkan/GPU acceleration is claimed. Large-model RAM requirements vary by device.
- Import currently accepts classic GPT-2 GGUF, not arbitrary Llama/Qwen models;
  rejected architectures get explicit diagnostics. Extending the interface does
  not require changing the UI generation loop, but requires model validation and
  tokenizer/chat-template work.
- GGUF export builds its output in memory, appropriate for Ember's small mobile
  training presets, not a general multi-gigabyte conversion tool. Local imported
  files use mmap on inference; headers are bounded to 64 MB.
- GGUF is inference-only; it does not restore Adam state or corpus. Use normal
  checkpoint/corpus exports to back those up.
- Browser UI has automated module/id/RPC contract tests, not a physical browser
  session. Optional external foragers/teacher APIs are not required or verified
  by this migration.

## Convert an actual saved checkpoint on the desktop

```sh
export EMBER_LLAMA_LIB="$PWD/native/build-llama/libember_llama.so"
python -m ember.gguf_export data/checkpoints/my-checkpoint.emb /tmp/my-model.gguf \
    --quantization q8_0 --validate-native
```

The converter reads weights/config from the real checkpoint and the exact
embedded tokenizer (or `--tokenizer my-checkpoint.tok.json`). It validates
shapes, required/unexpected tensors and tokenizer schema; a missing or old v3
tokenizer is an error, not an invitation to guess from vocab size. The JSON report
includes source step, tensor counts, dtype choices, fallback-to-F32 warnings and
successful native-load facts when `--validate-native` is requested.
