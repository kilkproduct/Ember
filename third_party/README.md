# Third-party sources

## llama.cpp — native inference engine

Ember's production text generation runs on [llama.cpp](https://github.com/ggml-org/llama.cpp).

* **Pinned revision:** tag `b11838`, commit `4231b23dc2c46661d01157aac5245bdcf03ce4b1`
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

To fetch the same verified pin into a different directory, use
`scripts/fetch-llama-cpp.sh --dir /path/to/llama.cpp` and pass that directory as
`-DEMBER_LLAMA_CPP_DIR=/path/to/llama.cpp`. Capability generation requires the
checksum-verified fetch stamp; an unversioned arbitrary checkout is not reported
as the pinned runtime.

### Why this revision

b11836 already contained real `qwen35`/`qwen35moe` gated-delta graphs. b11838 adds
the upstream timer overflow fix and behavior-check CI, without graph/API changes.
b11840's later UI/Hexagon changes are not needed for this CPU-only Android build.
See [the architecture audit](../docs/architecture-compatibility.md) for exact
hashes, limitations, source-derived capabilities and executed test evidence.

Own EMB4 training is a Llama-style RMSNorm/RoPE/SwiGLU decoder exported strictly
to the Llama graph. Foreign GGUF inference uses each upstream graph unchanged;
inference compatibility never makes foreign weights trainable in Ember.
