"""Tests for Ember v3 byte-level BPE tokenizer and SQLite corpus."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ember.corpus import Corpus
from ember.spec import (
    SPECIAL_BOS,
    SPECIAL_DOC,
    SPECIAL_EOS,
    SPECIAL_PAD,
    SPECIAL_UNK,
)
from ember.tokenizer import BPETokenizer, train_bpe

SAMPLE_TEXT = (
    "the cat sat on the mat. the cat likes the warm morning sun.\n"
    "singapore is an island city-state. people in singapore love hawker food! "
) * 6


def test_bpe_special_ids_and_roundtrip() -> None:
    tok = train_bpe([SAMPLE_TEXT], vocab_size=400)
    assert tok.pad_id == SPECIAL_PAD == 256
    assert tok.bos_id == SPECIAL_BOS == 257
    assert tok.eos_id == SPECIAL_EOS == 258
    assert tok.doc_id == SPECIAL_DOC == 259
    assert tok.unk_id == SPECIAL_UNK == 260
    assert 261 < tok.vocab_size <= 400

    for text in [
        "",
        " ",
        "   multiple   spaces\tand\n\nnewlines  ",
        "the cat sat on the mat in singapore!",
        "Unicode: Café, naïve, 灯台の記録, العربية, 🔥🎉🚀🎊",
    ]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert SPECIAL_UNK not in ids


def test_bpe_compression_efficiency() -> None:
    tok = train_bpe([SAMPLE_TEXT], vocab_size=450)
    phrase = "the cat sat on the mat in singapore."
    ids = tok.encode(phrase)
    assert len(ids) < len(phrase.encode("utf-8")), "Learned merges must compress repeated words"


def test_bpe_save_load_and_version_validation() -> None:
    tok = train_bpe([SAMPLE_TEXT], vocab_size=350)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tokenizer.json"
        tok.save(p)
        reloaded = BPETokenizer.load(p)
        assert reloaded.vocab_size == tok.vocab_size
        assert reloaded.encode("singapore sun 🔥") == tok.encode("singapore sun 🔥")

        bad = tok.to_dict()
        bad["version"] = 1
        with pytest.raises(ValueError, match="version"):
            BPETokenizer.from_dict(bad)


def test_sqlite_corpus_and_legacy_jsonl_preservation() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        # Pre-populate a legacy corpus.jsonl to verify user training material is preserved
        legacy_jsonl = tdir / "corpus.jsonl"
        legacy_docs = [
            {"id": "leg1", "title": "Legacy Note", "source": "user", "kind": "page", "weight": 2,
             "text": "Preserved user training document from legacy corpus."},
            {"id": "leg2", "title": "Legacy Lesson", "source": "taught", "kind": "lesson", "weight": 8,
             "text": "### Prompt:\nHi\n\n### Response:\nHello from preserved lesson!"},
        ]
        legacy_jsonl.write_text("\n".join(json.dumps(d) for d in legacy_docs) + "\n", encoding="utf-8")

        corpus = Corpus(tdir)
        assert corpus.count() == 2, "Existing corpus.jsonl documents must be automatically preserved"

        # Add, URL-update, and exact content dedupe
        d1 = corpus.add("Fresh astronomy notes about Jupiter and Saturn.", title="Planets", url="https://example.org/planets")
        assert not d1["merged"]
        d2 = corpus.add("Fresh astronomy notes about Jupiter, Saturn, and Uranus.", title="Planets v2", url="https://example.org/planets")
        assert d2["id"] == d1["id"]

        dup = corpus.add("Fresh astronomy notes about Jupiter, Saturn, and Uranus.", title="Copy")
        assert dup["merged"] is True

        summary = corpus.summary()
        assert summary["documents"] == 3
        hits = corpus.retrieve_passages("What notes mention Jupiter and Uranus?")
        assert any("Uranus" in h for h in hits)

        assert corpus.remove(d1["id"]) is True
        assert corpus.count() == 2
        corpus.close()
