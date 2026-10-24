# GGUF architecture compatibility overhaul

## Root cause and boundary

The reported `architecture 'qwen35' is not supported … supported dense architectures: llama, qwen2, qwen3`
was emitted by **Ember**, in `ember/gguf_export.py::_validate_gguf_file`, called by
`Trainer.import_gguf` / model inspection through Python RPC and Android's model
import. It was not an Arena attachment parser exception. The old preflight also
required dense Q/K/V/RoPE tensors and scalar attention metadata, even for models
with different graphs. Merely adding `qwen35` to that list would still be wrong.

Both the previous pin and the selected pin contain actual upstream Qwen3.5 and
Qwen3.5-MoE implementations (`src/models/qwen35{,moe}.cpp`), including recurrent
gated-delta layers. No upstream graph fork, architecture substitution, tensor
renaming, missing-layer filling, or weight/metadata rewriting has been added.

## Exact dependency selection

| Candidate | Exact commit | Decision |
|---|---|---|
| b11836 | `f5027fb21307eb264aed483242603d68c87b0fe3` | Old pin; already has Qwen3.5 graphs. |
| **b11838** | **`4231b23dc2c46661d01157aac5245bdcf03ce4b1`** | Selected reproducible published patch; timer overflow fix plus behavior CI, unchanged model APIs/graphs. |
| b11840 | `36e36e63a54fc8f85b95e9c499a022b597e478dd` | Later UI dependencies and Hexagon CEIL work do not improve this CPU Android path. |

Source: `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b11838`.
SHA-256: `1395bfc5668ac43ff59dec7d4ee21e103ee6bb91ca8d10d704256ed8dd65e6b0`.
`third_party/llama.cpp.version` and the fetch script pin/checksum the source;
CMake capability generation checks the fetch stamp and fails on parser drift.
This is a deliberately small revision upgrade, **not a claim that new graphs
were introduced between b11836 and b11838**.

## Runtime-derived evidence, not a support promise

- `scripts/generate-runtime-capabilities.py` parses the actual registry, model
  factory, recurrent/hybrid/diffusion classifiers, graph MoE calls, quantization
  layouts and upstream structural limits. It emits `ember/runtime_capabilities.json`
  and a native build header. CMake tracks the generator and its source inputs.
- `ember_llama_capabilities()` returns the compiled manifest; Python
  `LlamaCppEngine.capabilities()`, RPC `runtime.capabilities`, and JNI/Kotlin
  `LlamaInference.capabilities()` expose it. A stale library without this ABI
  reports an explicit rebuild/install mismatch rather than claiming readiness.
- The manifest has **164 registry entries (including upstream's unknown sentinel),
  162 factory implementations, 141 text candidates**. These are *candidates*, not
  141 verified model families. The complete per-entry table is machine-readable.
- Negative interface restrictions are explicit: embedding/BERT/T5-encoder,
  diffusion, auxiliary CLIP/audio, and EAGLE/DFLASH/Gemma4-assistant draft/target
  coupling. They are not portrayed as missing upstream graphs. Pangu-Embedded
  is a decoder, not excluded merely because its name contains “embed”.
- `docs/verification/architecture-host.json` overlays **executed** host evidence
  on every registry row, including untested rows, quantization scope, tokenizer
  scope, ABI caveats, fixture origins/checksums and genuine-model limitations.
  The compiled manifest intentionally contains no synthetic test certification.

Representative evidence:

| Family | Upstream graph | Executed evidence | Still unverified |
|---|---|---|---|
| Llama | yes | Native F32/Q8_0 fixtures; training/export/logit/tokenizer parity and lifecycle tests | Arbitrary pretrained models/devices/quants |
| Qwen2 / Qwen2.5 / Qwen3 | yes | Native dense F32/Q8_0 fixtures, Qwen3 independent head dimensions, template/control IDs, streaming/cancel/recovery | Genuine pretrained weights and all tokenizer/templates |
| Mamba | recurrent | Complete native F32/mixed-Q8_0 synthetic graph, multi-token decode/cancel/reload | Real Mamba checkpoints/mobile performance |
| Qwen3-MoE | expert routing | Complete native F32/mixed-Q8_0 synthetic expert tensors and routing, multi-token decode/cancel/reload | Real MoE weights/mobile memory/throughput |
| Qwen3.5 / Qwen3.5-MoE | hybrid gated-delta | Source/factory recognition; intentionally incomplete graph reaches upstream and is rejected before registration | **No genuine Qwen3.5 inference test** |
| RWKV, other SSM/hybrid/MoE families | see full manifest | Registry/factory/classification only | Model-specific load, tokenizer, generation and devices |
| T5 encoder-decoder | yes | Wrapper now evaluates encoder and decoder-start token instead of decoder-only assumption | No T5 weight execution test; encoder input must fit explicitly configured batch |
| Embedding/diffusion/auxiliary/draft-only | often yes | Explicit text-interface exclusions | Their missing application pipelines are not implemented |

## Validation and staging stages

1. Android ContentResolver streams bounded chunks to persistent private
   `model_staging`; storage, cancellation, byte/integrity and atomic finalize
   checks remain. No URI-as-path, procfs reopen, broad storage permission,
   evictable cache dependency or destructive model/data migration.
2. Generic GGUF preflight validates structure, upstream-derived type layouts,
   positive common geometry and upstream layer/expert bounds. It preserves
   per-layer arrays and architecture-specific SSM/MoE/RoPE metadata; no dense
   tensor inventory is imposed on foreign models. Recognized quant layout is
   not a CPU kernel/model compatibility guarantee.
3. A private temporary hardlink (or bounded copy fallback) is prepared before native validation; the validated inode is atomically renamed into the library only after success. Native upstream model load validates actual graph/tensors/quantization/tokenizer.
   Context allocation uses the requested context, microbatch, threads and mmap
   policy; errors retain upstream details. No fallback precision/context change.
4. **The same live validation handle** tokenizes a fixed non-private diagnostic
   prompt, evaluates it and generates one token. Only success permits library
   publication/selection. Failure/cancel leaves previous selection and corpus
   intact. No huge second model load is performed for this check.
5. `nativeValidated`, `textInferenceVerified` and `validationStage=first-token-verified`
   in the import result describe that attempt, not a permanently resident JNI
   context or a guarantee of continued inference for all prompts. Metadata-only
   inspection explicitly says native checks are pending. The handle is unloaded
   before Android JNI generation; normal generation uses the chosen settings.

File access, metadata/architecture recognition, missing interface, native model
load, context, tokenizer/prompt and first-token failures remain distinct in
messages/error names. No private prompt text is added to diagnostics. UI shows
first-token verification separately from model-load progress.

## Mobile, quantization and modality limits

- CPU-only, static llama/ggml inside `libember_llama.so`; intended ABIs
  `arm64-v8a` and `x86_64`. x86_64 Android disables AVX/AVX2/FMA requirements.
  Compiling/package-checking both ABIs is not physical-device inference proof.
- Existing explicit ceilings remain: context 8192, batch 2048, threads 8,
  microbatch <= batch, GPU layers zero. User settings are not silently reduced
  on import/selection. Upstream may pad allocation upwards, exposed as
  `allocatedCtx`; the logical requested context is preserved. RoPE extrapolation
  beyond training emits a quality warning; GPT-2 learned-position overrun rejects
  with a context-configuration error instead of silently clamping.
- F32 KV/state policy remains; no automatic weight/KV requantization. Known
  Ember exports retain their KV admission estimate. Foreign models use weight
  bytes plus reserve only, **not made-up dense KV estimates for recurrent/MoE**.
  Metadata reports unknown context working set. Conservative admission may reject
  a large mmap/MoE model; passing it is not a RAM/performance/OS-kill guarantee.
- Existing explicit prompt/sliding-window truncation warnings remain; import
  does not truncate tensors or silently shrink model precision/context.
- Multimodal **image/audio/video/projector integration is absent**. A text-capable
  component of a multimodal family does not enable other modalities.
- Only upstream built-in chat template formatting is used. Unsupported templates
  reject with actionable errors; arbitrary Jinja/template execution is absent.
- Pretrained inference is separate from training: EMB4 checkpoints, own tokenizer,
  corpus, Parquet ingestion, optimization/resume and strict Llama export remain.
  Foreign GGUFs do not become trainable merely because they can generate text.

## Executed verification and reproducibility

Host build: CMake Release, CPU, pinned source; **182 Python tests passed** (two
expected overflow warnings from the divergence-rejection test). Includes existing
EMB4/training/logit parity/Parquet/private-staging/cancel/switch tests and new
registry/quant layouts, real recurrent/MoE graphs, malformed metadata, wrong
Qwen3.5 graph rejection, first-token publication/cancel gating, same-model context reconfiguration and exact-private-inode validation, and actual native unsupported-template first-token rejection (not a mock).

```sh
scripts/fetch-llama-cpp.sh
python scripts/generate-runtime-capabilities.py --json ember/runtime_capabilities.json --check
cmake -S native/llama -B native/build-llama -DEMBER_BUILD_LLAMA_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/build-llama -j2
export EMBER_LLAMA_LIB="$PWD/native/build-llama/libember_llama.so"
OPENBLAS_NUM_THREADS=1 pytest -q tests/
python scripts/verify-runtime-architectures.py --output docs/verification/architecture-host.json
```

Optional genuine test: add `--model /private/model.gguf --source-url URL --license
'MODEL WEIGHT LICENSE' --context N` to the report script. It streams SHA-256,
verifies actual load/tokenizer/first and continued tokens, and does not commit
weights. Model source/license cannot be inferred from an architecture name.

Original `1a91b7eee80749ae824f504efc8155f2.gguf` was not supplied. Official
QwenLM/Qwen3.5 GitHub now redirects to Qwen3.8; its release list has no weight
assets and README links Hugging Face/ModelScope, outside this sandbox's permitted
hosts. **No genuine Qwen3.5 weight checksum/license or original-file success is
asserted. No physical Android device has been tested.**

## APK scope and remaining device verification

CI runs host/native/Python/HTTP/JVM regressions, assembles **only debug universal**
with both ABIs, verifies its APK signature and package contents (including the
capability manifest), and uploads only `Ember-universal-signed.apk` in artifact
`Ember-universal-signed`. It uses the existing cached debug/development keystore,
**not production signing credentials**. Cache loss can change that signing key.
No unsigned release, ABI split, baseline or instrumentation/test APK is built.

Android build/signature/package results must be read from the workflow run for
the delivered commit. Historical SAF emulator URI-grant setup remains failed;
it is not claimed fixed. Instrumentation would require another APK and is not
run under this task's universal-only APK restriction. Real SAF/provider/device
acceptance and genuine hybrid-model performance remain outstanding.
