# Independent original GGUF v2 writer

`gguf_v2_writer.py` is the **unmodified** upstream `gguf-py/gguf/gguf.py`
from llama.cpp commit `e394084166baac09e8ee9a08a4686f907f7e5291`
(2023-09; GGUF_VERSION=2). MIT license alongside.
Source: https://github.com/ggml-org/llama.cpp/blob/e394084166baac09e8ee9a08a4686f907f7e5291/gguf-py/gguf/gguf.py
SHA256: `0f2d600e97b42a913520bc37576a42e9f5c5b5c0c9301f32e7453c0f61bccbd5`.

Tests use this original writer to independently serialize a complete tiny Llama
model in its actual v2 layout, then inspect with current official gguf-py and
execute with native llama.cpp. No header bytes or version constants are patched.
The weights are deterministic synthetic graph fixtures, NOT pretrained weights.
No v2 conversion is exposed by the app; its own exporter remains v3 LE only.
