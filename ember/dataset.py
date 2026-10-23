"""Token streams, sequence sampling and batching.

Corpus text is encoded in a single bounded streaming pass into compact contiguous
NumPy buffers (int32 token IDs, int32 targets, int32 span indices), so the full
corpus text is never duplicated in memory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .spec import SPECIAL_BOS, SPECIAL_EOS, SPECIAL_PAD

IGNORE_INDEX = -100
LESSON_MARKER = "### Response:"


def _doc_hash_score(doc_id: str) -> int:
    h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


def normalize_split(value):
    value=str(value or "").strip().lower()
    if value in ("train","training"):return "train"
    if value in ("val","valid","validation","dev","development","eval","evaluation"):return "validation"
    if value in ("test","testing","heldout","holdout"):return "test"
    return ""


@dataclass
class Document:
    id: str
    text: str
    source: str = "manual"
    kind: str = "page"
    weight: float = 1.0
    title: str = ""
    split: str = ""

    def holdout(self, frac: float = 0.05) -> bool:
        """Deterministic per-document holdout predicate."""
        if self.split in ("validation","test"):return True
        if self.split=="train":return False
        if frac <= 0:
            return False
        return (_doc_hash_score(self.id) % 10000) < int(round(frac * 10000))


def compute_holdout_ids(
    meta_rows: Sequence[tuple[str, str]],
    holdout_frac: float = 0.05,
) -> set[str]:
    """Compute deterministic held-out validation document IDs from ``(id, kind)`` pairs.

    Guarantees:
    - Strictly disjoint from training documents.
    - Deterministic across runs for the same corpus.
    - When ``holdout_frac > 0`` and there are at least 2 documents, ensures at least
      one validation document is held out even for small corpora (preferring non-lesson
      documents so hand-taught lessons remain in the training set).
    """
    if holdout_frac <= 0 or len(meta_rows) < 2:
        return set()
    threshold = int(round(holdout_frac * 10000))
    non_lessons = [(doc_id, _doc_hash_score(doc_id)) for doc_id, kind in meta_rows if kind != "lesson"]
    pool = non_lessons if non_lessons else [(doc_id, _doc_hash_score(doc_id)) for doc_id, _ in meta_rows]
    val_ids = {doc_id for doc_id, score in pool if (score % 10000) < threshold}
    # Never hold out everything if pool happened to collide on low scores
    if len(val_ids) >= len(meta_rows):
        keep_id = max(pool, key=lambda item: item[1])[0]
        val_ids.discard(keep_id)
    if not val_ids and pool:
        best_id = min(pool, key=lambda item: item[1] % 10000)[0]
        val_ids.add(best_id)
    return val_ids


@dataclass
class DocSpan:
    start: int
    length: int
    source: str
    kind: str
    weight: float
    doc_id: str


@dataclass
class TokenStream:
    """Flat encoded corpus: contiguous int32 arrays."""

    ids: np.ndarray  # int32 [N]
    doc: np.ndarray  # int32 [N] index into spans
    targets: np.ndarray  # int32 [N] next-token id, IGNORE_INDEX where masked
    spans: list[DocSpan] = field(default_factory=list)
    truncated: bool = False
    sampled_documents: int = 0
    corpus_documents: int = 0
    sampled_chars: int = 0
    corpus_chars: int = 0
    source_tokens: dict[str, int] = field(default_factory=dict)
    source_boosts: dict[str, float] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.size)

    @property
    def loss_tokens(self) -> int:
        return int(np.count_nonzero(self.targets != IGNORE_INDEX))


class _GrowableInt32:
    """Amortized O(1) append buffer for int32 tokens without list-of-arrays overhead."""

    def __init__(self, initial_cap: int = 4096) -> None:
        self._buf = np.empty(max(64, initial_cap), dtype=np.int32)
        self.size = 0

    def extend_array(self, arr: np.ndarray) -> None:
        n = int(arr.size)
        if n == 0:
            return
        need = self.size + n
        if need > self._buf.size:
            new_cap = max(need, self._buf.size * 2)
            grown = np.empty(new_cap, dtype=np.int32)
            if self.size:
                grown[: self.size] = self._buf[: self.size]
            self._buf = grown
        self._buf[self.size : need] = arr
        self.size = need

    def extend_const(self, value: int, count: int) -> None:
        if count <= 0:
            return
        need = self.size + count
        if need > self._buf.size:
            new_cap = max(need, self._buf.size * 2)
            grown = np.empty(new_cap, dtype=np.int32)
            if self.size:
                grown[: self.size] = self._buf[: self.size]
            self._buf = grown
        self._buf[self.size : need] = int(value)
        self.size = need

    def finish(self) -> np.ndarray:
        return self._buf[: self.size].copy()


def _doc_tokens(tok, text: str, kind: str, mask_lesson_prompts: bool) -> tuple[list[int], int]:
    """Return ``(ids, n_prompt_tokens_to_mask)`` for one document body."""
    ids = tok.encode(text)
    mask = 0
    if kind == "lesson" and mask_lesson_prompts:
        idx = text.find(LESSON_MARKER)
        if idx >= 0:
            prefix = text[: idx + len(LESSON_MARKER)]
            mask = min(len(ids), len(tok.encode(prefix)))
    return ids, mask


def _compute_source_boosts(source_tokens: dict[str, int], balance: str) -> dict[str, float]:
    if balance not in ("gentle", "on", "soft", "sqrt", "equal") or not source_tokens:
        return {src: 1.0 for src in source_tokens}
    mean = sum(source_tokens.values()) / max(1, len(source_tokens))
    power = 0.75 if balance == "equal" else 0.5
    out: dict[str, float] = {}
    for src, n in source_tokens.items():
        out[src] = float(min(6.0, max(0.35, (mean / max(1, n)) ** power)))
    return out


def build_stream(
    documents: Iterable[Document],
    tok,
    *,
    max_tokens: int = 3_000_000,
    holdout_frac: float = 0.05,
    holdout_ids: set[str] | None = None,
    train: bool = True,
    mask_lesson_prompts: bool = False,
    balance: str = "off",
    progress=None,
    should_cancel=None,
) -> TokenStream:
    """Encode documents into a flat bounded token stream."""
    id_buf = _GrowableInt32()
    tgt_buf = _GrowableInt32()
    doc_buf = _GrowableInt32()
    spans: list[DocSpan] = []

    total = 0
    seen = 0
    kept = 0
    seen_chars = 0
    kept_chars = 0
    truncated = False
    source_tokens: dict[str, int] = {}

    # If caller gave a concrete sequence and no explicit holdout_ids, compute the
    # deterministic partition once from document metadata.
    if holdout_ids is None and isinstance(documents, Sequence):
        holdout_ids = compute_holdout_ids([(d.id, d.kind) for d in documents], holdout_frac)

    for doc in documents:
        text = doc.text or ""
        n_chars = len(text)
        seen += 1
        seen_chars += n_chars

        if progress and seen % 64 == 0:
            progress(seen, kept)
        if should_cancel is not None and seen % 32 == 0 and should_cancel():
            break

        if doc.split=="test":continue # final test data is never training or model-selection validation
        is_val = doc.split=="validation" or (doc.split!="train" and ((doc.id in holdout_ids) if holdout_ids is not None else doc.holdout(holdout_frac)))
        if train and is_val:
            continue
        if (not train) and (not is_val):
            continue

        if truncated:
            # Count remaining corpus metadata without tokenizing
            continue

        body, mask = _doc_tokens(tok, text, doc.kind, mask_lesson_prompts)
        if not body:
            continue

        seq_len = len(body) + 2  # <bos> ... <eos>
        if total + seq_len > max_tokens and total > 0:
            truncated = True
            continue
        if seq_len > max_tokens:
            body = body[: max(1, max_tokens - 2)]
            mask = min(mask, len(body))
            seq_len = len(body) + 2
            truncated = True

        arr = np.empty(seq_len, dtype=np.int32)
        arr[0] = SPECIAL_BOS
        arr[1:-1] = np.asarray(body, dtype=np.int32)
        arr[-1] = SPECIAL_EOS

        tgt = np.empty(seq_len, dtype=np.int32)
        tgt[:-1] = arr[1:]
        tgt[-1] = IGNORE_INDEX
        if mask > 0:
            # arr[0] has target arr[1] (first prompt token), up to arr[mask-1] -> arr[mask]
            tgt[:mask] = IGNORE_INDEX

        span_idx = len(spans)
        id_buf.extend_array(arr)
        tgt_buf.extend_array(tgt)
        doc_buf.extend_const(span_idx, seq_len)

        src = doc.source or "manual"
        source_tokens[src] = source_tokens.get(src, 0) + len(body)
        spans.append(
            DocSpan(
                start=total,
                length=seq_len,
                source=src,
                kind=doc.kind or "page",
                weight=float(doc.weight or 1.0),
                doc_id=doc.id,
            )
        )
        total += seq_len
        kept += 1
        kept_chars += n_chars

    boosts = _compute_source_boosts(source_tokens, balance)
    if boosts:
        for sp in spans:
            sp.weight = sp.weight * boosts.get(sp.source, 1.0)

    return TokenStream(
        ids=id_buf.finish(),
        doc=doc_buf.finish(),
        targets=tgt_buf.finish(),
        spans=spans,
        truncated=truncated,
        sampled_documents=kept,
        corpus_documents=seen,
        sampled_chars=kept_chars,
        corpus_chars=seen_chars,
        source_tokens=source_tokens,
        source_boosts=boosts,
    )


def build_train_val_streams(
    documents: Iterable[Document],
    tok,
    *,
    max_tokens: int = 3_000_000,
    max_val_tokens: int = 250_000,
    holdout_frac: float = 0.05,
    holdout_ids: set[str] | None = None,
    mask_lesson_prompts: bool = False,
    balance: str = "off",
    progress=None,
    should_cancel=None,
) -> tuple[TokenStream, TokenStream]:
    """Single-pass streaming builder producing both train and validation streams
    without iterating or tokenizing the corpus twice."""
    t_id, t_tgt, t_doc = _GrowableInt32(), _GrowableInt32(), _GrowableInt32()
    v_id, v_tgt, v_doc = _GrowableInt32(), _GrowableInt32(), _GrowableInt32()
    t_spans: list[DocSpan] = []
    v_spans: list[DocSpan] = []

    t_total = 0
    v_total = 0
    seen = 0
    seen_chars = 0
    t_kept = 0
    v_kept = 0
    t_chars = 0
    v_chars = 0
    truncated = False
    source_tokens: dict[str, int] = {}

    if holdout_ids is None and isinstance(documents, Sequence):
        holdout_ids = compute_holdout_ids([(d.id, d.kind) for d in documents], holdout_frac)

    for doc in documents:
        text = doc.text or ""
        n_chars = len(text)
        seen += 1
        seen_chars += n_chars

        if progress and seen % 32 == 0:
            progress(seen, t_kept + v_kept)
        if should_cancel is not None and seen % 32 == 0 and should_cancel():
            break

        if doc.split=="test":continue # final test data is never training or model-selection validation
        is_val = doc.split=="validation" or (doc.split!="train" and ((doc.id in holdout_ids) if holdout_ids is not None else doc.holdout(holdout_frac)))
        if is_val:
            if v_total >= max_val_tokens:
                continue
        else:
            if truncated:
                continue

        body, mask = _doc_tokens(tok, text, doc.kind, mask_lesson_prompts)
        if not body:
            continue

        budget = max_val_tokens if is_val else max_tokens
        cur_total = v_total if is_val else t_total
        seq_len = len(body) + 2
        if cur_total + seq_len > budget and cur_total > 0:
            if not is_val:
                truncated = True
            continue
        if seq_len > budget:
            body = body[: max(1, budget - 2)]
            mask = min(mask, len(body))
            seq_len = len(body) + 2
            if not is_val:
                truncated = True

        arr = np.empty(seq_len, dtype=np.int32)
        arr[0] = SPECIAL_BOS
        arr[1:-1] = np.asarray(body, dtype=np.int32)
        arr[-1] = SPECIAL_EOS

        tgt = np.empty(seq_len, dtype=np.int32)
        tgt[:-1] = arr[1:]
        tgt[-1] = IGNORE_INDEX
        if mask > 0:
            tgt[:mask] = IGNORE_INDEX

        src = doc.source or "manual"
        if is_val:
            span_idx = len(v_spans)
            v_id.extend_array(arr)
            v_tgt.extend_array(tgt)
            v_doc.extend_const(span_idx, seq_len)
            v_spans.append(DocSpan(v_total, seq_len, src, doc.kind or "page", 1.0, doc.id))
            v_total += seq_len
            v_kept += 1
            v_chars += n_chars
        else:
            span_idx = len(t_spans)
            t_id.extend_array(arr)
            t_tgt.extend_array(tgt)
            t_doc.extend_const(span_idx, seq_len)
            source_tokens[src] = source_tokens.get(src, 0) + len(body)
            t_spans.append(
                DocSpan(t_total, seq_len, src, doc.kind or "page", float(doc.weight or 1.0), doc.id)
            )
            t_total += seq_len
            t_kept += 1
            t_chars += n_chars

    boosts = _compute_source_boosts(source_tokens, balance)
    for sp in t_spans:
        sp.weight = sp.weight * boosts.get(sp.source, 1.0)

    train_stream = TokenStream(
        ids=t_id.finish(),
        doc=t_doc.finish(),
        targets=t_tgt.finish(),
        spans=t_spans,
        truncated=truncated,
        sampled_documents=t_kept,
        corpus_documents=seen,
        sampled_chars=t_chars,
        corpus_chars=seen_chars,
        source_tokens=source_tokens,
        source_boosts=boosts,
    )
    val_stream = TokenStream(
        ids=v_id.finish(),
        doc=v_doc.finish(),
        targets=v_tgt.finish(),
        spans=v_spans,
        truncated=False,
        sampled_documents=v_kept,
        corpus_documents=seen,
        sampled_chars=v_chars,
        corpus_chars=seen_chars,
    )
    return train_stream, val_stream


class Batcher:
    """Samples training windows out of a :class:`TokenStream`."""

    def __init__(
        self,
        stream: TokenStream,
        block_size: int,
        batch_size: int,
        seed: int = 0,
        sampler: str = "random",
        boundaries: str = "respect",
    ) -> None:
        self.stream = stream
        self.T = int(block_size)
        self.B = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        self.boundaries = boundaries
        self._cursor = 0
        self._order: np.ndarray | None = None
        n_spans = len(stream.spans)
        if n_spans:
            lengths = np.array([s.length for s in stream.spans], dtype=np.float64)
            weights = np.array([max(1e-6, s.weight) for s in stream.spans], dtype=np.float64)
            p = lengths * weights
            self.probs = p / p.sum()
        else:
            self.probs = np.zeros(0)

    def _next_shuffle_span(self) -> DocSpan:
        spans = self.stream.spans
        if self._order is None or self._cursor >= len(self._order):
            self._order = self.rng.permutation(len(spans))
            self._cursor = 0
        sp = spans[int(self._order[self._cursor])]
        self._cursor += 1
        return sp

    def batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(x, y, doc_ids)`` each shaped ``[B, T]``.

        When ``boundaries == 'respect'``, short documents stay isolated in their
        window (padded with ``<pad>`` and ``IGNORE_INDEX``) or contiguous chunks
        receive per-token ``doc_ids`` so attention cannot cross boundaries.
        """
        stream = self.stream
        if stream.n_tokens <= 1:
            raise ValueError("the training stream is empty — add fuel before firing")
        T = self.T
        x = np.full((self.B, T), SPECIAL_PAD, dtype=np.int32)
        y = np.full((self.B, T), IGNORE_INDEX, dtype=np.int32)
        d = np.full((self.B, T), -1, dtype=np.int32)

        for i in range(self.B):
            if self.boundaries == "respect" and stream.spans:
                if self.sampler in ("epoch", "shuffle"):
                    span = self._next_shuffle_span()
                else:
                    si = int(self.rng.choice(len(stream.spans), p=self.probs))
                    span = stream.spans[si]
                # Allow sampling window inside this span or spanning packed adjacent docs
                # with doc_ids masking keeping every document strictly self-contained.
                if span.length >= T:
                    max_off = span.length - T
                    start = span.start + (int(self.rng.integers(0, max_off + 1)) if max_off > 0 else 0)
                    end = start + T
                else:
                    start = span.start
                    end = min(stream.n_tokens, start + T)
            else:
                if stream.n_tokens <= T:
                    start = 0
                    end = stream.n_tokens
                elif self.sampler in ("epoch", "shuffle"):
                    stride = max(1, T)
                    n_win = max(1, (stream.n_tokens - T + stride - 1) // stride)
                    if self._order is None or self._cursor >= len(self._order):
                        self._order = self.rng.permutation(n_win)
                        self._cursor = 0
                    w = int(self._order[self._cursor])
                    self._cursor += 1
                    start = min(w * stride, stream.n_tokens - T)
                    end = start + T
                else:
                    start = int(self.rng.integers(0, stream.n_tokens - T + 1))
                    end = start + T

            take = end - start
            x[i, :take] = stream.ids[start:end]
            y[i, :take] = stream.targets[start:end]
            d[i, :take] = stream.doc[start:end]
            # Give padded tail distinct doc IDs per position so pad tokens never attend anywhere
            if take < T:
                d[i, take:] = np.arange(-1, -(T - take) - 1, -1, dtype=np.int32)
                if self.boundaries != "respect":
                    y[i, take:] = IGNORE_INDEX
            elif self.boundaries != "respect":
                # Flat stream mode allows cross-document attention; still never predict across <eos>-><bos>
                pass

        doc_ids_out = d if self.boundaries == "respect" else None
        return x, y, doc_ids_out


def split_documents(documents: Sequence[Document], holdout_frac: float = 0.05) -> tuple[list[Document], list[Document]]:
    """Partition documents into strictly disjoint ``(train, validation)`` lists."""
    holdout = compute_holdout_ids([(d.id, d.kind) for d in documents], holdout_frac)
    train, val = [], []
    for doc in documents:
        if doc.split=="test":continue
        is_val=doc.split=="validation" or (doc.split!="train" and doc.id in holdout)
        (val if is_val else train).append(doc)
    return train, val
