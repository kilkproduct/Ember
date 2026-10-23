# Third-party sources

## llama.cpp — native inference engine

Ember's production text generation runs on [llama.cpp](https://github.com/ggml-org/llama.cpp).

* **Pinned revision:** tag `b11836`, commit `f5027fb21307eb264aed483242603d68c87b0fe3`
* **Manifest (with SHA-256 and rationale):** [`llama.cpp.version`](./llama.cpp.version)
* **License:** MIT — full text in [`../licenses/LICENSE-llama.cpp`](../licenses/LICENSE-llama.cpp)
* **Fetched into:** `third_party/llama.cpp` (git-ignored; never vendored into the repository)

Fetch the pinned source (CI does this automatically):

```sh
scripts/fetch-llama-cpp.sh
```

The script downloads the release tarball from `codeload.github.com`, verifies its
SHA-256 against `llama.cpp.version`, and extracts it. A checksum mismatch aborts
the build — no prebuilt binaries are ever downloaded, and nothing is compiled
from an unpinned source tree.

To build against your own checkout instead:

```sh
cmake -S native/llama -B build -DEMBER_LLAMA_CPP_DIR=/path/to/llama.cpp
```

### Why llama.cpp

Ember trains a small decoder-only transformer. llama.cpp already implements that
exact computation graph as its `gpt2` architecture (`src/models/gpt2.cpp`):
learned positional embeddings, pre-LayerNorm with gain and bias, fused QKV with
bias, attention output projection with bias, MLP up/down with biases, tied output
embeddings, `1/sqrt(head_dim)` attention scaling — and `ggml`'s `GGML_OP_GELU`
is `0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x^2)))`, identical to Ember's
activation. Migration is therefore a tensor-name/layout mapping plus a GGUF
export, not a change of model mathematics. See
[`../docs/llama-cpp-migration.md`](../docs/llama-cpp-migration.md).
