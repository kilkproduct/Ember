"""Byte-level BPE tokenizer for Ember v3.

The vocabulary is 256 raw byte tokens + 5 specials + learned merges.  Every
merge is a concatenation of two existing tokens, so any UTF-8 text can always
be encoded (no unknown tokens) and decoding is a byte concatenation followed by
a UTF-8 decode.

The pre-tokenizer (:func:`ember.spec.pretokenize`) and the merge loop are
re-implemented byte-for-byte in ``native/src/bpe.cpp``; the two are compared by
``tests/python/test_interop.py`` and ``native/tests``.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable

from .spec import (
    BASE_VOCAB,
    FIRST_MERGE_ID,
    SPECIAL_BOS,
    SPECIAL_DOC,
    SPECIAL_EOS,
    SPECIAL_NAMES,
    SPECIAL_PAD,
    SPECIAL_TOKENS,
    SPECIAL_UNK,
    TOKENIZER_FORMAT,
    TOKENIZER_TYPE,
    TOKENIZER_VERSION,
    pretokenize,
)

__all__ = ["BPETokenizer", "train_bpe", "count_chunks"]


def _token_to_bytes(token: str) -> bytes:
    """Vocabulary entries are strings of code points < 256 (one per byte)."""
    return token.encode("latin-1")


def _bytes_to_token(raw: bytes) -> str:
    return raw.decode("latin-1")


def count_chunks(text: str) -> int:
    return len(pretokenize(text.encode("utf-8")))


def _chunk_counts(texts: Iterable[str], progress=None, total_hint: int = 0) -> Counter:
    counts: Counter = Counter()
    seen = 0
    for text in texts:
        if not text:
            continue
        for chunk in pretokenize(text.encode("utf-8")):
            counts[tuple(chunk[i : i + 1] for i in range(len(chunk)))] += 1
        seen += 1
        if progress and (seen % 200 == 0):
            progress(seen, total_hint)
    return counts


def _stats(counts: Counter) -> Counter:
    pairs: Counter = Counter()
    for word, freq in counts.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_pair(pair, counts: Counter) -> Counter:
    first, second = pair
    out: Counter = Counter()
    for word, freq in counts.items():
        new_word: list[bytes] = []
        i = 0
        n = len(word)
        while i < n:
            if i < n - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        out[tuple(new_word)] = freq
    return out


class BPETokenizer:
    """Deterministic byte-level BPE tokenizer."""

    def __init__(self, vocab: list[str], merges: list[list[str]]) -> None:
        self.vocab = list(vocab)
        self.merges = [list(m) for m in merges]
        self.token_to_id = {_token_to_bytes(t): i for i, t in enumerate(self.vocab)}
        self.ranks = {(_token_to_bytes(a), _token_to_bytes(b)): i for i, (a, b) in enumerate(self.merges)}
        self._cache: dict[bytes, tuple[int, ...]] = {}
        if len(self.vocab) < FIRST_MERGE_ID:
            raise ValueError("vocabulary is missing the base byte tokens")
        for name, idx in SPECIAL_TOKENS.items():
            if idx >= len(self.vocab):
                raise ValueError(f"vocabulary is missing the <{name}> token")

    # -- basics ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return SPECIAL_PAD

    @property
    def bos_id(self) -> int:
        return SPECIAL_BOS

    @property
    def eos_id(self) -> int:
        return SPECIAL_EOS

    @property
    def doc_id(self) -> int:
        return SPECIAL_DOC

    @property
    def unk_id(self) -> int:
        return SPECIAL_UNK

    def id_to_token(self, idx: int) -> str:
        return self.vocab[idx]

    def token_bytes(self, idx: int) -> bytes:
        return _token_to_bytes(self.vocab[idx])

    # -- encoding -------------------------------------------------------
    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word: list[bytes] = [chunk[i : i + 1] for i in range(len(chunk))]
        if len(word) > 1:
            while True:
                best = None
                best_rank = None
                for a, b in zip(word, word[1:]):
                    rank = self.ranks.get((a, b))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best = (a, b)
                if best is None:
                    break
                first, second = best
                new_word: list[bytes] = []
                i = 0
                n = len(word)
                while i < n:
                    if i < n - 1 and word[i] == first and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
        ids = tuple(self.token_to_id.get(sym, SPECIAL_UNK) for sym in word)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode_bytes(self, data: bytes) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(data):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.encode_bytes(text.encode("utf-8"))
        if add_bos:
            ids = [SPECIAL_BOS] + ids
        if add_eos:
            ids = ids + [SPECIAL_EOS]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.vocab):
                continue
            if skip_special and i >= BASE_VOCAB and i < FIRST_MERGE_ID:
                continue
            buf.extend(_token_to_bytes(self.vocab[i]))
        return buf.decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int], skip_special: bool = True) -> bytes:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.vocab):
                continue
            if skip_special and i >= BASE_VOCAB and i < FIRST_MERGE_ID:
                continue
            buf.extend(_token_to_bytes(self.vocab[i]))
        return bytes(buf)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "type": TOKENIZER_TYPE,
            "spec": "v3",
            "base": BASE_VOCAB,
            "specials": dict(SPECIAL_TOKENS),
            "vocab": list(self.vocab),
            "merges": [list(m) for m in self.merges],
        }

    def to_json(self) -> str:
        # ensure_ascii keeps the file pure ASCII, which is what the C++
        # reader (and Android's asset pipeline) expects.
        return json.dumps(self.to_dict(), ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        version = int(data.get("version", 0))
        if version != TOKENIZER_VERSION:
            raise ValueError(
                "tokenizer version %s is not supported by this build (expected %d)"
                % (data.get("version"), TOKENIZER_VERSION)
            )
        specials = data.get("specials") or {}
        for name, idx in SPECIAL_TOKENS.items():
            if int(specials.get(name, -1)) != idx:
                raise ValueError(f"tokenizer specials do not match the v3 layout (<{name}>)")
        return cls(list(data["vocab"]), [list(m) for m in data.get("merges", [])])

    @classmethod
    def from_json(cls, text: str) -> "BPETokenizer":
        return cls.from_dict(json.loads(text))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 2048,
    min_pair_count: int = 2,
    progress=None,
    total_hint: int = 0,
) -> BPETokenizer:
    """Learn a byte-level BPE vocabulary from ``texts``.

    ``texts`` is iterated once and consumed in bounded chunks: only the
    pre-tokenised chunk counts are held in memory (needed by the GPT-2 style
    merge loop), never the corpus itself.
    """
    vocab_size = max(FIRST_MERGE_ID + 1, int(vocab_size))
    counts = _chunk_counts(texts, progress=progress, total_hint=total_hint)
    if not counts:
        raise ValueError("cannot train a tokenizer on an empty corpus")

    vocab = [_bytes_to_token(bytes([i])) for i in range(BASE_VOCAB)]
    for name in ("pad", "bos", "eos", "doc", "unk"):
        vocab.append(f"<{name}>")
    merges: list[list[str]] = []

    target_merges = vocab_size - len(vocab)
    while len(merges) < target_merges:
        pairs = _stats(counts)
        if not pairs:
            break
        best_count = max(pairs.values())
        if best_count < min_pair_count:
            break
        # Deterministic tie-break: lowest (count desc, pair) wins.
        best = min((p for p, c in pairs.items() if c == best_count), key=lambda p: (p[0], p[1]))
        merges.append([_bytes_to_token(best[0]), _bytes_to_token(best[1])])
        vocab.append(_bytes_to_token(best[0] + best[1]))
        counts = _merge_pair(best, counts)
        if progress:
            progress(len(merges), target_merges)

    return BPETokenizer(vocab, merges)
