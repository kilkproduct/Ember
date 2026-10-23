"""Deterministic cross-language verification between Python training/reference
and the compiled C++17 native inference library (`libember_infer.so`)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ember import EmberBackend, NativeInferenceBridge, reference_generate
from ember.model import GPT, ModelConfig, save_model
from ember.tokenizer import train_bpe


def _require_native() -> NativeInferenceBridge:
    bridge = NativeInferenceBridge()
    assert bridge.available, (
        "Compiled native library libember_infer.so must exist! "
        "Build native/CMakeLists.txt before running interop tests."
    )
    return bridge


def test_tokenizer_exact_parity_unicode_emoji_whitespace_boundaries() -> None:
    corpus = [
        "Alice was beginning to get very tired of sitting by her sister on the bank!\n\n",
        "Café au bord du fleuve — 灯台の記録 — emoji celebration 🔥🎉🚀🎊✨\t\t  ",
        "Code & symbols: fn main() { let x = a[0] + b.c('hi'); // 100% ok! }\r\n",
    ] * 8
    tok = train_bpe(corpus, vocab_size=520)
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=24, vocab_size=tok.vocab_size)
    gpt = GPT(cfg, seed=5)

    cpp = _require_native()
    try:
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "m.emb"
            tp = Path(td) / "t.json"
            save_model(mp, cfg, gpt.params)
            tok.save(tp)
            cpp.load(mp, tp)

            assert cpp.vocab_size == tok.vocab_size
            edge_strings = [
                "",
                " ",
                "    ",
                "\n",
                "\r\n\t  leading   and   trailing   \t\n",
                "punctuation!!! ??? --- ... ((([[[{{{}}}]]])))",
                "Café naïve résumé Zürich",
                "日本語のテキストと灯台の記録",
                "Emoji sequence: 🔥🎉🚀🎊✨🌍💡",
                "Mixed: [r/singapore] Hello 世界! 123_456 += 78.9% 🔥\nNext line.",
            ]
            for s in edge_strings:
                py_ids = tok.encode(s)
                cpp_ids = cpp.encode(s)
                assert py_ids == cpp_ids, f"Token ID mismatch on {s!r}: py={py_ids} vs cpp={cpp_ids}"
                assert cpp.decode(cpp_ids) == s, f"UTF-8 decode mismatch on {s!r}"
    finally:
        cpp.close()


def test_forward_logits_greedy_and_sliding_context_parity_after_training() -> None:
    with tempfile.TemporaryDirectory() as td:
        be = EmberBackend(td)
        be.dispatch("corpus.addPack", {"pack": "starter"})
        be.dispatch("corpus.addPack", {"pack": "topic"})
        be.dispatch("lesson.add", {"prompt": "Question?", "response": "Exact trained answer 🔥!", "weight": 10})

        # Small context window (24) intentionally forces sliding-window rollover during generation!
        be.dispatch("vessel.cast", {"nLayer": 3, "nHead": 4, "nEmbd": 64, "blockSize": 24, "vocabSize": 550})
        be.dispatch("train.sync", {"steps": 30, "batch": 4, "lr": 1e-3, "every": 15, "validateEvery": 15})
        mpath, tpath = be.trainer._sync_active_files()

        cpp = _require_native()
        try:
            cpp.load(mpath, tpath)

            # 1. Compare full [T, V] float32 logits across multiple sequence lengths up to block_size
            for text in ["A", "Alice looked up at the stars!", "Café 灯台 🔥🎉 test sequence"]:
                ids = ([be.trainer.tokenizer.bos_id] + be.trainer.tokenizer.encode(text))[:24]
                py_logits = be.trainer.model.forward_logits(np.asarray([ids], dtype=np.int32))[0]
                cpp_logits = cpp.forward_logits(ids)
                max_diff = float(np.max(np.abs(py_logits - cpp_logits)))
                assert max_diff < 1e-5, f"Max float32 logit diff {max_diff} exceeded 1e-5 tolerance"

            # 2. Greedy decode parity across short prompt, empty prompt, AND over-length prompt (> block_size)
            prompts = [
                "",
                "Alice was",
                # Prompt longer than block_size=24 -> tests right-truncation + sliding KV cache rebuild!
                "Stars appear to move across the night sky from east to west because the Earth rotates every day",
            ]
            for prompt in prompts:
                py_g = reference_generate(
                    be.trainer.model,
                    be.trainer.tokenizer,
                    prompt,
                    max_new_tokens=36,
                    temperature=0.0,
                    top_k=1,
                    repetition_penalty=1.15,
                    seed=101,
                    stop_on_eos=False,
                )
                cpp_g = cpp.generate(
                    prompt,
                    max_new_tokens=36,
                    temperature=0.0,
                    top_k=1,
                    repetition_penalty=1.15,
                    seed=101,
                    stop_on_eos=False,
                )
                assert py_g.token_ids == cpp_g.token_ids, f"Greedy decode diverged for prompt={prompt!r}"
                assert py_g.text == cpp_g.text

            # 3. Seeded stochastic sampling parity (SplitMix64 + temperature + top-k + top-p + repetition penalty)
            py_s = reference_generate(
                be.trainer.model,
                be.trainer.tokenizer,
                "Sunlight warms",
                max_new_tokens=30,
                temperature=0.85,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.2,
                seed=987654321,
                stop_on_eos=False,
            )
            cpp_s = cpp.generate(
                "Sunlight warms",
                max_new_tokens=30,
                temperature=0.85,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.2,
                seed=987654321,
                stop_on_eos=False,
            )
            assert py_s.token_ids == cpp_s.token_ids

            # 4. Run compiled standalone C++ test binary directly against Python-produced .emb + .tok.json
            native_bin = Path(__file__).resolve().parent.parent / "native" / "build" / "test_native"
            if native_bin.exists():
                proc = subprocess.run([str(native_bin), str(mpath), str(tpath)], capture_output=True, text=True, check=False)
                assert proc.returncode == 0, f"C++ binary test failed: {proc.stderr}\n{proc.stdout}"
        finally:
            cpp.close()
            be.close()


def test_native_malformed_and_vocab_mismatch_handling() -> None:
    cpp = _require_native()
    try:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            tok_small = train_bpe(["hello world hello world"] * 5, vocab_size=300)
            tok_big = train_bpe(["hello world a b c d e f"] * 20, vocab_size=340)
            cfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=tok_small.vocab_size)
            gpt = GPT(cfg, seed=3)

            mp = tdir / "valid.emb"
            tp_ok = tdir / "ok.json"
            tp_mismatch = tdir / "mismatch.json"
            save_model(mp, cfg, gpt.params)
            tok_small.save(tp_ok)
            tok_big.save(tp_mismatch)

            # Mismatched vocab sizes between .emb and tokenizer must fail cleanly
            with pytest.raises(ValueError, match="vocab_size"):
                cpp.load(mp, tp_mismatch)

            # Corrupt .emb checksum must fail cleanly without crashing C++
            bad_mp = tdir / "corrupt.emb"
            raw = bytearray(mp.read_bytes())
            raw[64] ^= 0x55
            bad_mp.write_bytes(bytes(raw))
            with pytest.raises(ValueError, match="checksum"):
                cpp.load(bad_mp, tp_ok)

            # Legacy JSON brain passed to native engine must give clear legacy error
            leg_mp = tdir / "legacy.json"
            leg_mp.write_text('{"format": "ember-brain-v2"}', encoding="utf-8")
            with pytest.raises(ValueError, match="Legacy JSON brain"):
                cpp.load(leg_mp, tp_ok)

            # Engine must still cleanly load valid files afterwards (repeated reload & cleanup)
            for _ in range(5):
                cpp.load(mp, tp_ok)
                res = cpp.generate("hello", max_new_tokens=8, temperature=0.0, top_k=1)
                assert res.new_tokens == 8
    finally:
        cpp.close()
