"""GPT-2 compatible pre-tokenizer and GGUF vocabulary mapping.

llama.cpp executes Ember models with its built-in ``gpt2`` architecture, whose
tokenizer is **byte-level BPE with the classic GPT-2 pre-tokenizer regex**::

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

and whose vocabulary strings are stored in the GPT-2 *byte encoder* alphabet
(``bytes_to_unicode``), not as raw bytes.

Ember trains on the device, so the Python side must chunk text exactly the way
llama.cpp will at inference time; otherwise training and inference disagree
about token boundaries.  This module is that single source of truth:

* :func:`pretokenize_gpt2` reproduces the regex above without depending on the
  third-party ``regex`` package (which is not available on Android),
* :func:`bytes_to_unicode` is the canonical GPT-2 byte encoder, and
* :func:`gguf_vocab_entries` converts an Ember :class:`~ember.tokenizer.BPETokenizer`
  into the ``(tokens, token_types, merges)`` arrays stored in a GGUF file.

Chunk boundaries produced here are verified against llama.cpp's own tokenizer in
``tests/test_llama_inference.py`` (native parity test).
"""

from __future__ import annotations

import unicodedata
from typing import Callable, Iterable, Sequence

# GGUF tokenizer.ggml.token_type values (gguf-py/gguf/constants.py::TokenType)
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

# Ember's special ids (see ember/spec.py) are exported as CONTROL tokens so the
# BPE pre-tokenizer can never emit them as text and the sampler can suppress
# them explicitly.
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def bytes_to_unicode() -> dict[int, str]:
    """Canonical GPT-2 byte encoder (mirrors gguf-py/gguf/vocab.py)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def encode_token_bytes(raw: bytes) -> str:
    """Map raw token bytes into the GPT-2 byte-encoder alphabet."""
    return "".join(_BYTE_TO_UNICODE[b] for b in raw)


def decode_token_string(text: str) -> bytes:
    """Inverse of :func:`encode_token_bytes`."""
    return bytes(_UNICODE_TO_BYTE.get(ch, ord("?")) for ch in text)


def latin1_to_bytes(token: str) -> bytes:
    """Ember vocabulary entries are latin-1 strings: one code point per byte."""
    return token.encode("latin-1")


# llama.cpp's whitespace set (src/unicode-data.cpp::unicode_set_whitespace).
# Matching this table exactly is what makes Python chunking agree with
# llama_tokenize; Python's str.isspace() differs on a few code points.
WHITESPACE_CODEPOINTS = frozenset((
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
))


def _is_space(ch: str) -> bool:
    return ord(ch) in WHITESPACE_CODEPOINTS


def _is_letter(ch: str) -> bool:
    """\\p{L}: Lu, Ll, Lt, Lm, Lo — Python's str.isalpha() is exactly this set."""
    return ch.isalpha()


def _is_number(ch: str) -> bool:
    """\\p{N}: Nd, Nl, No."""
    return ch.isnumeric()


def _is_other(ch: str) -> bool:
    """[^\\s\\p{L}\\p{N}]"""
    return not (_is_space(ch) or _is_letter(ch) or _is_number(ch))


def pretokenize_gpt2(text: str) -> list[str]:
    """Split ``text`` into GPT-2 BPE chunks exactly like llama.cpp does.

    This mirrors ``unicode_regex_split_custom_gpt2`` in ``src/unicode.cpp`` of the
    pinned llama.cpp revision, rule by rule, rather than re-deriving the pattern
    from the regex text:

    1. ``'s|'t|'re|'ve|'m|'ll|'d``                    contractions
    2. ``<space>?\\p{L}+``                              optional *literal* space + letters
    3. ``<space>?\\p{N}+``                              optional literal space + numbers
    4. ``<space>?[^\\s\\p{L}\\p{N}]+``                 optional literal space + other
    5. ``\\s+(?!\\S)``  → implemented as: run length > 1
       **and** not at end of text → take ``run - 1``
    6. ``\\s+``                                        otherwise take the whole run
    7. no match → the single character becomes its own chunk

    Note the space in rules 2–4 is a literal U+0020, not ``\\s``: a tab or
    newline before a word is emitted as its own chunk (rule 6).
    """
    out: list[str] = []
    n = len(text)
    pos = 0

    def flags_letter(i: int) -> bool:
        return i < n and _is_letter(text[i])

    def flags_number(i: int) -> bool:
        return i < n and _is_number(text[i])

    def flags_space(i: int) -> bool:
        return i < n and _is_space(text[i])

    def flags_other(i: int) -> bool:
        # A position past the end has no flags at all (llama.cpp returns an empty
        # flag word there), so it must not satisfy the "other characters" rule.
        return i < n and _is_other(text[i])

    while pos < n:
        ch = text[pos]

        # 1) contractions: 's 't 'm 'd (2 chars) and 're 've 'll (3 chars)
        if ch == "'" and pos + 1 < n:
            nxt = text[pos + 1]
            if nxt in ("s", "t", "m", "d"):
                out.append(text[pos : pos + 2])
                pos += 2
                continue
            if pos + 2 < n:
                pair = nxt + text[pos + 2]
                if pair in ("re", "ve", "ll"):
                    out.append(text[pos : pos + 3])
                    pos += 3
                    continue

        # Rules 2-4 look at the *next* character when the current one is a space.
        probe = pos + 1 if ch == " " else pos

        # 2) <space>?\p{L}+
        if flags_letter(probe):
            start = pos
            i = probe
            while flags_letter(i):
                i += 1
            out.append(text[start:i])
            pos = i
            continue

        # 3) <space>?\p{N}+
        if flags_number(probe):
            start = pos
            i = probe
            while flags_number(i):
                i += 1
            out.append(text[start:i])
            pos = i
            continue

        # 4) <space>?[^\s\p{L}\p{N}]+
        if flags_other(probe):
            start = pos
            i = probe
            while flags_other(i):
                i += 1
            out.append(text[start:i])
            pos = i
            continue

        # 5)/6) whitespace runs
        num_ws = 0
        while flags_space(pos + num_ws):
            num_ws += 1

        if num_ws > 1 and pos + num_ws < n:
            # \s+(?!\S): keep the last whitespace character with the next chunk.
            out.append(text[pos : pos + num_ws - 1])
            pos += num_ws - 1
            continue

        if num_ws > 0:
            out.append(text[pos : pos + num_ws])
            pos += num_ws
            continue

        # 7) no rule matched: the character is its own chunk.
        out.append(ch)
        pos += 1

    return out


def gguf_vocab_entries(
    vocab: Sequence[str],
    merges: Sequence[Sequence[str]],
    *,
    special_ids: Iterable[int] = (),
) -> tuple[list[str], list[int], list[str]]:
    """Convert an Ember byte-level vocabulary into GGUF ``gpt2`` tokenizer arrays.

    ``vocab`` holds latin-1 strings (one code point per raw byte) and ``merges``
    holds pairs of those strings, in rank order.  Returns
    ``(tokens, token_types, merges)`` with tokens expressed in the GPT-2
    byte-encoder alphabet and merges as ``"a b"`` strings — the exact shape
    ``tokenizer.ggml.tokens`` / ``.token_type`` / ``.merges`` expect.
    """
    specials = {int(s) for s in special_ids}
    tokens: list[str] = []
    types: list[int] = []
    for idx, entry in enumerate(vocab):
        if idx in specials:
            tokens.append(entry if entry.startswith("<") else f"<{entry}>")
            types.append(TOKEN_TYPE_CONTROL)
        else:
            tokens.append(encode_token_bytes(latin1_to_bytes(entry)))
            types.append(TOKEN_TYPE_NORMAL)

    gguf_merges: list[str] = []
    for rank, pair in enumerate(merges):
        if len(pair) != 2:
            raise ValueError(f"merge #{rank} must have exactly two parts, got {list(pair)!r}")
        gguf_merges.append(" ".join(encode_token_bytes(latin1_to_bytes(part)) for part in pair))

    return tokens, types, gguf_merges
