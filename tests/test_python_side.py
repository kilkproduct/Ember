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
from ember.tokenizer import ByteBpeTokenizer, EmberTokenizer

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


def test_bpe_roundtrip_and_determinism() -> None:
    tok = ByteBpeTokenizer()
    tok.train([TEXT * 10], vocab_size=800)
    for s in ["", "hello, world!", "café 🔥 中文", "unseen αβγ", "a  b\n\nc\td"]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, f"BPE round-trip failed for {s!r}"
        assert tok.unk_id not in ids, "byte BPE must never emit <unk>"
    again = ByteBpeTokenizer()
    again.train([TEXT * 10], vocab_size=800)
    assert again.merges == tok.merges, "BPE training must be deterministic"
    assert tok.vocab_size == 261 + len(tok.merges) <= 800


def test_bpe_golden_merges_match_js() -> None:
    # The exact merge list web/bpe.js learns on this corpus (same golden
    # asserted in tests/bpe.test.mjs) — JS/Python tokenizer parity.
    gold = (
        "the cat sat on the mat. the cat likes the warm sun. "
        "a small cat sleeps on the warm mat in the sun."
    )
    tok = ByteBpeTokenizer()
    tok.train([gold], vocab_size=280)
    assert [list(p) for p in tok.merges] == [
        [102, 121], [109, 106], [121, 262], [37, 120], [37, 263], [37, 104],
        [266, 261], [37, 114], [37, 116], [37, 124], [102, 119], [122, 115],
        [264, 272], [268, 261], [269, 115], [270, 271], [276, 114],
    ], f"merges drifted from the JS implementation: {tok.merges}"


def test_bpe_save_load_and_tamper() -> None:
    tok = ByteBpeTokenizer()
    tok.train([TEXT * 6], vocab_size=700)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bpe.json"
        tok.save(path)
        back = ByteBpeTokenizer.load(path)
        assert back.encode("people in singapore") == tok.encode("people in singapore")
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))
        data["id_to_token"][270] = "forgery"
        path.write_text(_json.dumps(data), encoding="utf-8")
        try:
            ByteBpeTokenizer.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered BPE vocab must be rejected")


def test_legacy_unk_is_visible() -> None:
    tok = EmberTokenizer()
    tok.train([TEXT], vocab_size=1500)
    ids = tok.encode("cat")
    assert tok.decode(ids + [tok.unk_id]) == "cat�"


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_tokenizer_greedy_match()
    test_tokenizer_save_load()
    test_corpus()
    test_bpe_roundtrip_and_determinism()
    test_bpe_golden_merges_match_js()
    test_bpe_save_load_and_tamper()
    test_legacy_unk_is_visible()
    print("python-side tests: OK")
