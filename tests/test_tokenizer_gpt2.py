"""Tokenizer tests for the GPT-2 pre-tokenizer and the GGUF vocabulary mapping.

llama.cpp tokenizes Ember models with its ``gpt-2`` pre-tokenizer, so Ember's
Python side must chunk text identically — otherwise training and inference
disagree about token boundaries. ``tests/test_llama_inference.py`` compares these
chunks against ``llama_tokenize`` directly; this module pins the rules in pure
Python so a regression is caught even without a native build.
"""

from __future__ import annotations

import pytest

from ember.gpt2_tokenizer import (
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    bytes_to_unicode,
    decode_token_string,
    encode_token_bytes,
    gguf_vocab_entries,
    pretokenize_gpt2,
)
from ember.spec import SPECIAL_BOS, SPECIAL_DOC, SPECIAL_EOS, SPECIAL_PAD, SPECIAL_UNK
from ember.tokenizer import BPETokenizer, train_bpe

CORPUS = [
    "The tide rises and the tide falls. The tide rises again at dusk.",
    "Stars rise in the east and set in the west every single night.",
    "It doesn't matter whether we'll arrive; they're already there.",
    "Flour, water and salt make bread — kneading aligns the gluten network.",
    "灯台の記録とالعربية и русский: multilingual fuel for a tiny model.",
    "Emoji 🔥🎉🚀 and symbols ±≠≈ 100% (parentheses) [brackets] {braces}.",
]


def test_pretokenizer_contraction_rules() -> None:
    # The GPT-2 regex owns contractions: the apostrophe joins the suffix.
    assert pretokenize_gpt2("don't") == ["don", "'t"]
    assert pretokenize_gpt2("it's") == ["it", "'s"]
    assert pretokenize_gpt2("we'll") == ["we", "'ll"]
    assert pretokenize_gpt2("they're") == ["they", "'re"]
    assert pretokenize_gpt2("I've") == ["I", "'ve"]
    assert pretokenize_gpt2("he'd") == ["he", "'d"]
    assert pretokenize_gpt2("won't") == ["won", "'t"]
    # An apostrophe that is not a contraction stays punctuation.
    assert pretokenize_gpt2("the dogs' bowls") == ["the", " dogs", "'", " bowls"]


def test_pretokenizer_whitespace_rules() -> None:
    # A single space prefixes the next chunk; a run leaves its last space to do so.
    assert pretokenize_gpt2("hello world") == ["hello", " world"]
    assert pretokenize_gpt2("a  b") == ["a", " ", " b"]
    assert pretokenize_gpt2("a   b") == ["a", "  ", " b"]
    assert pretokenize_gpt2("trailing   ") == ["trailing", "   "]
    assert pretokenize_gpt2("\n\n") == ["\n\n"]
    assert pretokenize_gpt2("line\nnext") == ["line", "\n", "next"]
    assert pretokenize_gpt2("\tmixed \v whitespace") == ["\t", "mixed", " \v", " whitespace"]
    # A non-breaking space is Unicode whitespace (Zs), not a letter.
    assert pretokenize_gpt2("a\u00a0b") == ["a", "\u00a0", "b"]


def test_pretokenizer_unicode_classes() -> None:
    assert pretokenize_gpt2("Café au lait") == ["Café", " au", " lait"]
    assert pretokenize_gpt2("灯台の記録") == ["灯台の記録"]
    assert pretokenize_gpt2("1234 56") == ["1234", " 56"]
    assert pretokenize_gpt2("x2 y10") == ["x", "2", " y", "10"]
    # Letters and numbers are separate classes in the GPT-2 regex.
    assert pretokenize_gpt2("abc123") == ["abc", "123"]
    assert pretokenize_gpt2("🔥🎉") == ["🔥🎉"]
    assert pretokenize_gpt2("a—b") == ["a", "—", "b"]
    assert pretokenize_gpt2("50% off!") == ["50", "%", " off", "!"]


def test_pretokenizer_is_lossless() -> None:
    for text in CORPUS + ["", " ", "  ", "a", "...", "\r\n\t ", "𝄞 clef", "e\u0301 combining"]:
        assert "".join(pretokenize_gpt2(text)) == text, f"chunking lost characters in {text!r}"


def test_byte_encoder_is_a_bijection_over_all_256_bytes() -> None:
    mapping = bytes_to_unicode()
    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256
    for b in range(256):
        assert encode_token_bytes(bytes([b])) == mapping[b]
        assert decode_token_string(mapping[b]) == bytes([b])
    # Whitespace and control bytes must be mapped away from themselves.
    assert encode_token_bytes(b" ") == "Ġ"
    assert encode_token_bytes(b"\n") != "\n"
    assert encode_token_bytes(b"\x00") != "\x00"
    # Printable ASCII maps to itself.
    assert encode_token_bytes(b"hello") == "hello"


def test_gguf_vocab_entries_encode_tokens_and_mark_specials() -> None:
    tok = train_bpe(CORPUS, vocab_size=400)
    tokens, types, merges = gguf_vocab_entries(
        tok.vocab, tok.merges, special_ids=(SPECIAL_PAD, SPECIAL_BOS, SPECIAL_EOS, SPECIAL_DOC, SPECIAL_UNK)
    )
    assert len(tokens) == len(types) == tok.vocab_size
    assert len(merges) == len(tok.merges)

    # The 256 byte tokens are the GPT-2 alphabet, in id order.
    mapping = bytes_to_unicode()
    for b in range(256):
        assert tokens[b] == mapping[b]
        assert types[b] == TOKEN_TYPE_NORMAL

    # Ember's specials keep their literal names and become CONTROL tokens, which
    # is how the native engine knows to suppress them during sampling.
    assert tokens[SPECIAL_PAD:SPECIAL_UNK + 1] == ["<pad>", "<bos>", "<eos>", "<doc>", "<unk>"]
    assert types[SPECIAL_PAD:SPECIAL_UNK + 1] == [TOKEN_TYPE_CONTROL] * 5

    # Merges are "a b" strings in the same alphabet, in rank order.
    for rank, pair in enumerate(tok.merges):
        expected = " ".join(encode_token_bytes(p.encode("latin-1")) for p in pair)
        assert merges[rank] == expected
        assert len(merges[rank].split(" ")) == 2


def test_gguf_vocab_rejects_malformed_merges() -> None:
    with pytest.raises(ValueError, match="two parts"):
        gguf_vocab_entries(["a"], [["a", "b", "c"]])


def test_tokenizer_encode_decode_roundtrip_with_the_gpt2_chunker() -> None:
    tok = train_bpe(CORPUS, vocab_size=420)
    assert tok.vocab_size <= 420
    cases = [
        "",
        " ",
        "   multiple   spaces\tand\n\nnewlines  ",
        "the cat sat on the mat in singapore!",
        "It doesn't matter — they're coming, we'll see; I've waited.",
        "Unicode: Café, naïve, 灯台の記録, العربية, русский",
        "Emoji 🔥🎉🚀🎊 and symbols ±≠≈ 100%",
        "Mixed: [r/singapore] Hello 世界! 123_456 += 78.9% 🔥\nNext line.",
        "a\u00a0non-breaking space",
        "\x00 embedded NUL and \x1f control bytes",
    ]
    for text in cases:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round trip failed for {text!r}"
        assert SPECIAL_UNK not in ids, f"byte-level BPE must never need <unk> for {text!r}"


def test_encode_bytes_matches_encode_for_valid_utf8() -> None:
    tok = train_bpe(CORPUS, vocab_size=400)
    for text in CORPUS:
        assert tok.encode_bytes(text.encode("utf-8")) == tok.encode(text)


def test_special_token_ids_are_unchanged_by_the_migration() -> None:
    """The id layout is the contract between training data and every GGUF export."""
    tok = train_bpe(CORPUS, vocab_size=300)
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.doc_id, tok.unk_id) == (256, 257, 258, 259, 260)
    assert tok.vocab[:256] == [bytes([b]).decode("latin-1") for b in range(256)]
    assert tok.vocab[256:261] == ["<pad>", "<bos>", "<eos>", "<doc>", "<unk>"]


def test_saved_tokenizer_roundtrips_and_rejects_other_versions(tmp_path) -> None:
    tok = train_bpe(CORPUS, vocab_size=340)
    path = tmp_path / "tok.json"
    tok.save(path)
    reloaded = BPETokenizer.load(path)
    assert reloaded.vocab_size == tok.vocab_size
    for text in CORPUS:
        assert reloaded.encode(text) == tok.encode(text)

    bad = tok.to_dict()
    bad["version"] = 2
    with pytest.raises(ValueError, match="version"):
        BPETokenizer.from_dict(bad)

    wrong_specials = tok.to_dict()
    wrong_specials["specials"]["eos"] = 999
    with pytest.raises(ValueError, match="specials"):
        BPETokenizer.from_dict(wrong_specials)
