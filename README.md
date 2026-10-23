# Ember — local training, llama.cpp inference

Ember is a native Kotlin/Jetpack Compose Android language foundry. Python/NumPy
(embedded with Chaquopy) trains small Llama 3–style models from a persistent SQLite
corpus. **Every production generation uses llama.cpp** from local GGUF weights.
There is no cloud inference, Python generator fallback, or server process in the APK.

## On Android

1. Install the `Ember-debug.apk` from **Actions → Build APK → Ember-debug**.
2. Either add fuel in Fuel, **Cast** in Vessel, and **Fire** to train; or use
   **Voice → Import GGUF model** to copy a compatible model into app-private storage.
3. In Voice select the model, choose the context/sampling settings, and Generate.
   Output streams as tokens arrive. **Stop** retains partial output.
4. **Unload** releases native model memory; files remain on disk. The next Generate
   reloads the selected file. Selection survives restart.
5. **Export GGUF** opens Android's file picker. F32 is lossless; Q8_0 quantizes the
   seven projection matrices per transformer block. Checkpoints/corpus exports remain
   separate: a GGUF file is for inference, not a backup of your training corpus.

This build accepts **Llama-architecture GGUF with Ember’s classic `gpt-2` byte-BPE tokenizer**,
including models exported by Ember. A `.gguf` extension alone is not compatibility:
metadata, dimensions, tensors and successful native loading matter. It does not
accept arbitrary Llama 3 tokenizer files or scaled-RoPE models. Architecture and tokenizer are independent: this is a small Llama-style decoder, not Meta pretrained weights.

After installation and local model setup, generation requires no internet.
INTERNET permission remains for **optional**, explicit foraging, teachers and
GitHub sync; these are not used as a fallback for local inference.

## Reproducible native dependency

Upstream: <https://github.com/ggml-org/llama.cpp>, MIT.
Pinned release **b11836**, commit `f5027fb21307eb264aed483242603d68c87b0fe3`.
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
