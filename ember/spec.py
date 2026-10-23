"""Shared Ember model + tokenizer specification, version 3.

This module is the single source of truth for everything the Python trainer
(``ember.model``) and the native C++ inference engine (``native/``) must agree
on:

* the on-disk weight format (``.emb``, magic ``EMB3``),
* the tensor names, shapes and row-major float32 ordering,
* the tokenizer contract (byte-level BPE, special ids, pre-tokenizer rules),
* the numeric details (LayerNorm epsilon, GELU variant, attention scaling).

Both implementations read this file's *behaviour*, not its code: the C++ side
re-implements the same rules in ``native/src/model.cpp`` / ``bpe.cpp``.  The
constants and layout below are mirrored in ``native/include/ember/spec.hpp``
and cross-checked by ``tests/python/test_interop.py``.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# File format
# --------------------------------------------------------------------------

FORMAT_MAGIC = b"EMB4"
FORMAT_VERSION = 4
DTYPE_FLOAT32 = 0
DTYPE_NAMES = {DTYPE_FLOAT32: "float32"}
DTYPE_ITEMSIZE = {DTYPE_FLOAT32: 4}

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

BASE_VOCAB = 256  # ids 0..255 are the raw UTF-8 bytes
SPECIAL_PAD = 256
SPECIAL_BOS = 257
SPECIAL_EOS = 258
SPECIAL_DOC = 259
SPECIAL_UNK = 260
FIRST_MERGE_ID = 261

SPECIAL_TOKENS = {
    "pad": SPECIAL_PAD,
    "bos": SPECIAL_BOS,
    "eos": SPECIAL_EOS,
    "doc": SPECIAL_DOC,
    "unk": SPECIAL_UNK,
}
SPECIAL_NAMES = {v: k for k, v in SPECIAL_TOKENS.items()}

TOKENIZER_TYPE = "byte-bpe"
TOKENIZER_FORMAT = "ember-bpe"
TOKENIZER_VERSION = 4

# Byte classes used by the pre-tokenizer.  These are byte-level on purpose:
# the base vocabulary is the 256 possible UTF-8 bytes, so Unicode and emoji
# need no tables in C++ and both implementations split text identically.
CLASS_SPACE = 0
CLASS_WORD = 1
CLASS_PUNCT = 2

SPACE_BYTES = frozenset((0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D))


def byte_class(b: int) -> int:
    """Classify one byte.  Mirrored by ``ember::byteClass`` in C++."""
    if b in SPACE_BYTES:
        return CLASS_SPACE
    # Letters, digits, underscore, and every byte >= 0x80 (so multi-byte
    # UTF-8 sequences and emoji stay inside one chunk).
    if (
        (0x30 <= b <= 0x39)
        or (0x41 <= b <= 0x5A)
        or (0x61 <= b <= 0x7A)
        or b == 0x5F
        or b >= 0x80
    ):
        return CLASS_WORD
    return CLASS_PUNCT


def pretokenize(data: bytes) -> list[bytes]:
    """Split UTF-8 ``data`` into BPE chunks.

    Rules (identical in C++, see ``ember::pretokenize``):

    * a maximal run of whitespace bytes is split so that its **last** byte
      starts the following chunk (``"  hi"`` -> ``[" ", " hi"]``); trailing
      whitespace becomes one chunk per byte,
    * a maximal run of word bytes (letters/digits/``_``/any byte >= 0x80)
      is one chunk,
    * a maximal run of punctuation bytes is one chunk.
    """
    out: list[bytes] = []
    i = 0
    n = len(data)
    while i < n:
        cls = byte_class(data[i])
        if cls == CLASS_SPACE:
            j = i
            while j < n and byte_class(data[j]) == CLASS_SPACE:
                j += 1
            if j < n:
                for k in range(i, j - 1):
                    out.append(data[k : k + 1])
                start = j - 1
                cls2 = byte_class(data[j])
                k = j
                while k < n and byte_class(data[k]) == cls2:
                    k += 1
                out.append(data[start:k])
                i = k
            else:
                for k in range(i, j):
                    out.append(data[k : k + 1])
                i = j
        else:
            j = i
            while j < n and byte_class(data[j]) == cls:
                j += 1
            out.append(data[i:j])
            i = j
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

ARCH = "ember-llama"
NORM_EPS = 1e-5
ACTIVATION = "swiglu"
POSITIONAL = "rope"
DEFAULT_FFN_MULT = 4
MASK_NEG = -1e9  # additive attention mask value for blocked positions

# Presets exposed by the Vessel screen.  Sizes are chosen for a phone: the
# default "phone" vessel is ~0.6M parameters at vocab 2048.
PRESETS = {
    "phone": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "vocab_size": 1500, "dropout": 0.0},
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "vocab_size": 2000, "dropout": 0.0},
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128, "vocab_size": 3000, "dropout": 0.0},
    "medium": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 192, "vocab_size": 4000, "dropout": 0.0},
    "tablet": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256, "vocab_size": 4096, "dropout": 0.0},
    "desktop": {"n_layer": 8, "n_head": 8, "n_embd": 256, "block_size": 384, "vocab_size": 8192, "dropout": 0.0},
}


def ffn_dim(n_embd: int, ffn_mult: float) -> int:
    return max(1, int(round(n_embd * float(ffn_mult or DEFAULT_FFN_MULT))))


def tensor_specs(cfg) -> list[tuple[str, list[int]]]:
    c, f, kv = cfg.n_embd, cfg.ffn, cfg.n_kv_head * cfg.head_dim
    specs = [("wte", [cfg.vocab_size, c])]
    for i in range(cfg.n_layer):
        p = f"blk.{i}."
        specs += [(p+"attn_norm.weight", [c]), (p+"attn_q.weight", [c,c]),
                  (p+"attn_k.weight", [c,kv]), (p+"attn_v.weight", [c,kv]),
                  (p+"attn_output.weight", [c,c]), (p+"ffn_norm.weight", [c]),
                  (p+"ffn_gate.weight", [c,f]), (p+"ffn_up.weight", [c,f]),
                  (p+"ffn_down.weight", [f,c])]
    specs += [("output_norm.weight", [c])]
    if not cfg.tie_embeddings: specs += [("lm_head.weight", [cfg.vocab_size,c])]
    return specs


def estimate_params(n_layer: int, n_head: int, n_embd: int, block_size: int, vocab_size: int, ffn_mult: float = 4.0) -> int:
    """Number of trainable float32 parameters."""
    f = ffn_dim(n_embd, ffn_mult)
    embed = vocab_size * n_embd + block_size * n_embd
    per_block = (
        4 * n_embd * n_embd  # qkv + attn projection
        + 2 * n_embd * f  # mlp
        + 3 * n_embd  # qkv bias
        + n_embd  # attn proj bias
        + f  # mlp fc bias
        + n_embd  # mlp proj bias
        + 4 * n_embd  # two layernorms
    )
    total = embed + n_layer * per_block + 2 * n_embd
    return int(total)


def estimate_train_bytes(params: int) -> int:
    """Weights + gradients + Adam first/second moments (float32)."""
    return int(params) * 16


def gelu(x):
    """tanh-approximated GELU (the only activation in Ember v3)."""
    import numpy as np

    c = 0.7978845608028654  # sqrt(2 / pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def gelu_back(x, dy):
    """d/dx gelu(x) * dy, using the activation input ``x``."""
    import numpy as np

    c = 0.7978845608028654
    t = np.tanh(c * (x + 0.044715 * x * x * x))
    inner = 1.0 + 0.134145 * x * x  # 3 * 0.044715
    dt = (1.0 - t * t) * c * inner
    return dy * (0.5 * (1.0 + t) + 0.5 * x * dt)
