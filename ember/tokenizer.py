"""Wordpiece-ish tokenizer: characters + frequent words/substrings from the corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"]


class EmberTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL)}
        self.id_to_token: list[str] = list(SPECIAL)
        self._max_tok_len = 1

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def doc_id(self) -> int:
        return 4

    def _add(self, tok: str) -> None:
        if tok not in self.token_to_id:
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)
            if len(tok) > self._max_tok_len:
                self._max_tok_len = len(tok)

    def train(self, texts: list[str], vocab_size: int = 4000) -> None:
        """Build a vocab from raw documents. Always keeps every seen character."""
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL)}
        self.id_to_token = list(SPECIAL)
        self._max_tok_len = 1
        chars: set[str] = set()
        word_counts: Counter[str] = Counter()
        for text in texts:
            for ch in text:
                if ch not in ("\x00",):
                    chars.add(ch)
            for tok in TOKEN_RE.findall(text):
                if tok.strip() == "":
                    # keep a single space token; collapse longer whitespace later
                    word_counts[" "] += 1
                else:
                    word_counts[tok] += 1
        for ch in sorted(chars):
            self._add(ch)
        # Remaining slots go to the most common multi-char tokens
        remaining = max(0, vocab_size - self.vocab_size)
        for tok, _n in word_counts.most_common():
            if remaining <= 0:
                break
            if tok in self.token_to_id:
                continue
            if len(tok) < 2:
                continue
            self._add(tok)
            remaining -= 1

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for tok in TOKEN_RE.findall(text):
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
                continue
            # Greedy longest-match inside the regex token so vocab pieces are
            # reused inside unseen words, falling back to single characters
            # (then <unk>) so nothing is silently lost. Mirrors web/tokenizer.js.
            i = 0
            while i < len(tok):
                hit_id = None
                hit_len = 0
                for length in range(min(self._max_tok_len, len(tok) - i), 1, -1):
                    cand = tok[i : i + length]
                    if cand in self.token_to_id:
                        hit_id = self.token_to_id[cand]
                        hit_len = length
                        break
                if hit_id is not None:
                    ids.append(hit_id)
                    i += hit_len
                else:
                    ids.append(self.token_to_id.get(tok[i], self.unk_id))
                    i += 1
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.doc_id}
        parts: list[str] = []
        for i in ids:
            if i in skip:
                if i == self.doc_id:
                    parts.append("\n\n")
                continue
            if 0 <= i < len(self.id_to_token):
                tok = self.id_to_token[i]
                if tok == "<unk>":
                    # visible replacement char — mirrors web/tokenizer.js;
                    # silently dropping unknowns hid tokenizer gaps
                    parts.append("\ufffd")
                    continue
                parts.append(tok)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id_to_token": self.id_to_token}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmberTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.id_to_token = list(data["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        tok._max_tok_len = max((len(t) for t in tok.id_to_token), default=1)
        return tok


# ---------------------------------------------------------------------------
# Byte-level BPE — the Python mirror of web/bpe.js (Stage D). Same special
# ids (pad 0, unk 1, bos 2, eos 3, doc 4), byte b at id 5+b, merge i at
# 261+i, same pre-tokenization, same deterministic tie-break (highest
# count, then smallest (a, b) pair) and the same token length /
# whitespace-run guards, so JS and Python learn identical merges from
# the same corpus (verified by a golden test in tests/test_python_side.py
# and tests/bpe.test.mjs).
# ---------------------------------------------------------------------------

BPE_CHUNK_RE = re.compile(r" ?\w+| ?[^\w\s]|\s+", re.UNICODE)
_FIRST_MERGE_ID = 261
_MAX_TOKEN_BYTES = 48
_MAX_WS_TOKEN_BYTES = 8
_TRAIN_CHAR_BUDGET = 2_000_000
_WS_BYTES = frozenset((9, 10, 11, 12, 13, 32))


def _bytes_to_unicode() -> list[str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    out = [""] * 256
    for b, c in zip(bs, cs):
        out[b] = chr(c)
    return out


BYTE_TO_UNI = _bytes_to_unicode()


def _derive_bpe_vocab(merges: list[tuple[int, int]]) -> tuple[list[str], list[bytes]]:
    token_bytes: list[bytes] = [b""] * (_FIRST_MERGE_ID + len(merges))
    token_bytes[1] = "�".encode("utf-8")
    for byte in range(256):
        token_bytes[5 + byte] = bytes((byte,))
    for i, pair in enumerate(merges):
        if len(pair) != 2:
            raise ValueError(f"BPE merge {i} is not a pair")
        a, b = pair
        limit = _FIRST_MERGE_ID + i
        if not (5 <= a < limit and 5 <= b < limit):
            raise ValueError(f"BPE merge {i} {pair} references an invalid token id")
        token_bytes[_FIRST_MERGE_ID + i] = token_bytes[a] + token_bytes[b]
    id_to_token = [
        SPECIAL[i] if i < 5 else "".join(BYTE_TO_UNI[b] for b in tb)
        for i, tb in enumerate(token_bytes)
    ]
    return id_to_token, token_bytes


class ByteBpeTokenizer:
    """Byte-level BPE. Mirrors web/bpe.js (same ids, merges, strings)."""

    type = "byte-bpe"

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self._finalize()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def doc_id(self) -> int:
        return 4

    def _finalize(self) -> None:
        self.id_to_token, self._token_bytes = _derive_bpe_vocab(self.merges)
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._chunk_cache: dict[str, list[int]] = {}

    def train(self, texts: list[str], vocab_size: int = 1500) -> None:
        self.merges = []
        target = max(0, vocab_size - _FIRST_MERGE_ID)
        if target == 0:
            self._finalize()
            return
        chunk_freq: Counter[str] = Counter()
        budget = _TRAIN_CHAR_BUDGET
        done = False
        for text in texts:
            if done:
                break
            for raw in BPE_CHUNK_RE.findall(text):
                for i in range(0, len(raw), 64):
                    chunk_freq[raw[i : i + 64]] += 1
                budget -= len(raw)
                if budget <= 0:
                    done = True
                    break
        chunks: list[list[int]] = []
        freqs: list[int] = []
        for s, n in chunk_freq.items():
            data = s.encode("utf-8")
            if not data:
                continue
            chunks.append([5 + b for b in data])
            freqs.append(n)
        pair_count: Counter[tuple[int, int]] = Counter()
        pair_chunks: dict[tuple[int, int], set[int]] = {}

        def bump(ids: list[int], ci: int, delta: int) -> None:
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_count[pair] += delta
                if delta > 0:
                    pair_chunks.setdefault(pair, set()).add(ci)
                elif pair in pair_chunks:
                    pair_chunks[pair].discard(ci)

        for ci, ids in enumerate(chunks):
            bump(ids, ci, freqs[ci])
        tok_len = [1] * (_FIRST_MERGE_ID + target)
        tok_ws = [False] * (_FIRST_MERGE_ID + target)
        for byte in range(256):
            tok_ws[5 + byte] = byte in _WS_BYTES
        for m in range(target):
            best_pair: tuple[int, int] | None = None
            best_count = 0
            for pair, count in pair_count.items():
                if count < 2:
                    continue
                a, b = pair
                length = tok_len[a] + tok_len[b]
                if length > _MAX_TOKEN_BYTES:
                    continue
                if tok_ws[a] and tok_ws[b] and length > _MAX_WS_TOKEN_BYTES:
                    continue
                if count > best_count or (count == best_count and (best_pair is None or pair < best_pair)):
                    best_count = count
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            new_id = _FIRST_MERGE_ID + m
            self.merges.append(best_pair)
            tok_len[new_id] = tok_len[a] + tok_len[b]
            tok_ws[new_id] = tok_ws[a] and tok_ws[b]
            for ci in list(pair_chunks.get(best_pair, ())):
                old = chunks[ci]
                bump(old, ci, -freqs[ci])
                merged: list[int] = []
                i = 0
                while i < len(old):
                    if i + 1 < len(old) and old[i] == a and old[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(old[i])
                        i += 1
                chunks[ci] = merged
                bump(merged, ci, freqs[ci])
            pair_count.pop(best_pair, None)
            pair_chunks.pop(best_pair, None)
        self._finalize()

    def _encode_chunk(self, s: str) -> list[int]:
        ids = [5 + b for b in s.encode("utf-8")]
        while len(ids) >= 2:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(ids) - 1):
                r = self._ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            ids = ids[:best_idx] + [_FIRST_MERGE_ID + best_rank] + ids[best_idx + 2 :]
        return ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_special else []
        for chunk in BPE_CHUNK_RE.findall(text):
            part = self._chunk_cache.get(chunk)
            if part is None:
                part = self._encode_chunk(chunk)
                if len(chunk) <= 128:
                    if len(self._chunk_cache) > 8192:
                        self._chunk_cache.clear()
                    self._chunk_cache[chunk] = part
            ids.extend(part)
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {0, 2, 3, 4}
        data = b"".join(self._token_bytes[i] for i in ids if i not in skip and 0 <= i < len(self._token_bytes))
        return data.decode("utf-8", "replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "byte-bpe",
                    "version": 1,
                    "merges": [list(p) for p in self.merges],
                    "id_to_token": self.id_to_token,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteBpeTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte-bpe":
            raise ValueError("not a byte-bpe tokenizer file")
        tok = cls()
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._finalize()
        if data.get("id_to_token") is not None and list(data["id_to_token"]) != tok.id_to_token:
            raise ValueError("BPE vocabulary does not match its merges")
        return tok
