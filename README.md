# Ember — local training, llama.cpp inference

> **0.7 engineering overhaul:** [architecture, compatibility, measurements and limits](docs/overhaul.md) · [reproducible Android verification](docs/device-verification.md).

Ember is a native Kotlin/Jetpack Compose Android language foundry. Python/NumPy
(embedded with Chaquopy) trains small Llama 3–style models from a persistent SQLite
corpus. **Every production generation uses llama.cpp** from local GGUF weights.
There is no cloud inference, Python generator fallback, or server process in the APK.

## GGUF format and inspection

GGUF **v2/v3 little-endian** parsing and native generation are tested. v3
big-endian metadata can be inspected but is explicitly unsupported for inference
by the pinned C++ reader on Android. Model library **Details** shows declared
metadata and compatibility separately from actual native readiness.
See [the v2/v3 overhaul and current test evidence](docs/gguf-format-overhaul.md).

## On Android

1. Install `Ember-universal-signed.apk` from **Actions → Build APK → Ember-universal-signed** (development signed, not a production release).
2. Either add text in Data, **Cast** in Vessel, and **Fire** to train; or use
   **Chat → Model library → Import GGUF**. Android streams the selected document into persistent private storage; native load and one-token generation must succeed before registration.
3. In Chat select the model, choose the context/sampling settings, and Generate.
   Output streams as tokens arrive. **Stop** retains partial output.
4. **Unload** releases native model memory; files remain on disk. The next Generate
   reloads the selected file. Selection survives restart.
5. **Export GGUF** opens Android's file picker. F32 is lossless; Q8_0 quantizes the
   seven projection matrices per transformer block. Checkpoints/corpus exports remain
   separate: a GGUF file is for inference, not a backup of your training corpus.

GGUF architecture recognition now comes from the **compiled upstream registry and
model factory**, not Ember's old Llama/Qwen2/Qwen3 whitelist. Qwen3.5/Qwen3.5-MoE
hybrid graphs exist in this runtime; they are **not verified with genuine weights**.
Complete synthetic native **Llama, Qwen2/Qwen3, Mamba recurrent and Qwen3-MoE**
graphs are exercised in F32/mixed Q8_0. That is not proof every pretrained family,
quantization, tokenizer or phone will work. Upstream loading/context/tokenizer
and first-token validation gate each import; metadata inspection alone is not
readiness. No image/audio/video/projector pipeline or arbitrary Jinja engine is
provided. See the [compatibility overhaul and full evidence matrix](docs/architecture-compatibility.md).
Own training still uses Ember's classic GPT-2 byte BPE and unscaled Llama-style
RoPE; that is a small trainable decoder, not Meta pretrained weights.

After installation and local model setup, generation requires no internet.
INTERNET permission remains for **optional**, explicit foraging, teachers and
GitHub sync; these are not used as a fallback for local inference.

## Reproducible native dependency

Upstream: <https://github.com/ggml-org/llama.cpp>, MIT.
Pinned release **b11838**, commit `4231b23dc2c46661d01157aac5245bdcf03ce4b1`.
[`third_party/llama.cpp.version`](third_party/llama.cpp.version) records the source
URL, SHA-256 and selection rationale. No prebuilt native binaries are downloaded.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt cmake
scripts/build-native.sh                 # fetch/checksum/build pinned source
export EMBER_LLAMA_LIB="$PWD/native/build-llama/libember_llama.so"
pytest -v tests/
python scripts/make_test_gguf.py /tmp/ember-test.gguf --steps 4
native/build-llama/test_llama_engine /tmp/ember-test.gguf
./run.sh                               # optional desktop/web companion, port 8000
```

The fetch step needs internet **at build time**, not during inference. Source and
build products are ignored by Git. Android needs Java 17, Gradle 8.7, NDK
26.1.10909125, CMake 3.22.1, SDK 34; CI installs the tools and fetches the pinned
source before `gradle testDebugUnitTest assembleDebug` in `android/`.
ABIs: `arm64-v8a` and `x86_64`; minimum Android API 24.

## Train an existing corpus from the command line

```sh
python -m ember.train --data ./data --preset phone --steps 200 --seed 42 \
  --max-train-tokens 100000 --tokenizer-chars 200000 \
  --export model.gguf --quantization q8_0
```

The same training pipeline runs in the APK. The command-line environment is
optional, not required for normal Android use. Preparation limits are explicit
and do not delete documents. See the migration guide for JSONL import and resume.

## Architecture and format boundary

See [migration and audit](docs/llama-cpp-migration.md) for the tensor mapping,
precision limits, tests and unverified device checks.

- **EMB4** `.emb` + tokenizer JSON + Adam state: training checkpoints, retained on disk.
- `active.gguf`: atomic export of trained weights, loaded by llama.cpp.
- Tokenizer schema **v4**: GPT-2-compatible Unicode pre-tokenization; the old v3
  byte-class chunker cannot be silently reinterpreted. Recast/retrain old vessels.
- Existing source corpora, lessons and datasets are **not reset or deleted**.
- Legacy JavaScript weights and old custom-engine weights are not production
  inference files. Import/export diagnostics identify the mismatch explicitly.

The removed `libember_infer` forward pass, tokenizer and sampler are not compiled
or packaged. NumPy's training forward/backward implementation remains as the
numerical reference; the old custom generator and sampler are removed entirely.
