"""Stdlib-only tests for the Python tokenizer + corpus.

Run from the repo root:  python3 tests/test_python_side.py
(the PyTorch model in ember/model.py needs `pip install torch` and is not
exercised here; the shipped app trains in the browser via web/engine.js.)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ember.corpus import Corpus
from ember.tokenizer import EmberTokenizer

TEXT = (
    "the cat sat on the mat. the cat likes the sun. "
    "singapore is an island city. people in singapore like chicken rice. "
) * 4


def test_tokenizer_roundtrip() -> None:
    tok = EmberTokenizer()
    tok.train([TEXT], vocab_size=1500)
    s = "the cat sat on the mat in singapore."
    assert tok.decode(tok.encode(s)) == s, "round-trip must be exact"


def test_tokenizer_greedy_match() -> None:
    tok = EmberTokenizer()
    tok.train([TEXT], vocab_size=1500)
    ids = tok.encode("singaporeans")
    assert tok.decode(ids) == "singaporeans"
    # "singapore" is a vocab piece, so this must beat 1 token per character
    assert len(ids) < len("singaporeans"), f"expected piece reuse, got {len(ids)} tokens"


def test_tokenizer_save_load() -> None:
    tok = EmberTokenizer()
    tok.train([TEXT], vocab_size=1500)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tok.json"
        tok.save(path)
        back = EmberTokenizer.load(path)
    assert back.id_to_token == tok.id_to_token
    s = "people in singapore like chicken rice."
    assert back.encode(s) == tok.encode(s)


def test_corpus() -> None:
    with tempfile.TemporaryDirectory() as td:
        corpus = Corpus(Path(td) / "corpus.jsonl")
        corpus.add("hello world", title="a")
        corpus.add("second doc", title="b", weight=3)
        corpus.add("hello world v2", title="a2", url="http://x")
        corpus.add("hello world v3", title="a3", url="http://x")  # dedupe by url
        st = corpus.stats()
        assert st["documents"] == 3, st
        blob = corpus.weighted_blob()
        assert blob.count("second doc") == 3  # weight repeats
        # reload from disk
        again = Corpus(Path(td) / "corpus.jsonl")
        assert again.stats()["documents"] == 3


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_tokenizer_greedy_match()
    test_tokenizer_save_load()
    test_corpus()
    print("python-side tests: OK")
