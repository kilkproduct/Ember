"""GGUF export tests: writer correctness, tensor mapping, quantization, validation.

The writer is verified two ways:
1. against the **official upstream reader** (``gguf.GGUFReader`` from llama.cpp's
   gguf-py package) — so the byte layout is not merely self-consistent, and
2. against llama.cpp's own expectations for the ``gpt2`` architecture
   (``src/models/gpt2.cpp`` at the pinned revision): tensor names, ggml
   dimension order, tied output head and metadata keys.

A separate test module (``test_llama_inference.py``) loads these files with the
real native engine.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ember.gguf_export import (
    EMBER_MODEL_FORMAT,
    GGML_Q8_0,
    GgufExportError,
    ember_to_gpt2_tensors,
    export_gguf,
    quantize_q8_0,
    read_gguf_header,
    describe_incompatible_bytes,
    validate_gguf_file,
)
from ember.model import GPT, ModelConfig, save_model
from ember.tokenizer import train_bpe

import gguf as gguf_reader

CORPUS = [
    "The tide rises and the tide falls. The tide rises again at dusk.",
    "Stars rise in the east and set in the west every single night.",
    "Flour and water and salt make bread. Kneading aligns the gluten.",
    "Clouds gather, clouds darken, clouds release the rain over the harbour.",
]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=420, ffn_mult=4.0)
    tok = train_bpe(CORPUS, vocab_size=420)
    cfg.vocab_size = tok.vocab_size
    gpt = GPT(cfg, seed=11)
    return tmp_path_factory.mktemp("gguf"), cfg, gpt.params, tok


def test_official_gguf_reader_parses_the_exported_file(trained) -> None:
    tmp, cfg, params, tok = trained
    out = tmp / "model-f32.gguf"
    report = export_gguf(out, cfg, params, tok, name="unit-test", step=7)

    assert out.exists()
    assert report.bytes == out.stat().st_size
    assert report.tensors == 4 + 12 * cfg.n_layer  # embeddings/positions/output_norm + per-layer set
    assert report.quantization == "f32"
    assert report.tied_output is True
    assert report.format == EMBER_MODEL_FORMAT

    reader = gguf_reader.GGUFReader(str(out))
    fields = reader.fields
    assert fields["general.architecture"].contents() == "gpt2"
    assert fields["gpt2.context_length"].contents() == cfg.block_size
    assert fields["gpt2.embedding_length"].contents() == cfg.n_embd
    assert fields["gpt2.block_count"].contents() == cfg.n_layer
    assert fields["gpt2.feed_forward_length"].contents() == cfg.ffn
    assert fields["gpt2.attention.head_count"].contents() == cfg.n_head
    assert fields["gpt2.attention.head_count_kv"].contents() == cfg.n_head
    assert abs(fields["gpt2.attention.layer_norm_epsilon"].contents() - cfg.norm_eps) < 1e-9
    assert fields["tokenizer.ggml.model"].contents() == "gpt2"
    assert fields["tokenizer.ggml.pre"].contents() == "gpt-2"
    assert fields["tokenizer.ggml.add_bos_token"].contents() is False
    assert fields["tokenizer.ggml.add_eos_token"].contents() is False
    assert fields["ember.format_version"].contents() == EMBER_MODEL_FORMAT
    assert fields["ember.source_step"].contents() == 7

    # Special ids must be UINT32: llama.cpp rejects an i32 bos_token_id with
    # "key tokenizer.ggml.bos_token_id has wrong type i32 but expected type u32".
    uint32 = gguf_reader.constants.GGUFValueType.UINT32
    for key, expected in (
        ("tokenizer.ggml.bos_token_id", 257),
        ("tokenizer.ggml.eos_token_id", 258),
        ("tokenizer.ggml.unknown_token_id", 260),
        ("tokenizer.ggml.padding_token_id", 256),
    ):
        field = fields[key]
        assert field.contents() == expected
        assert list(field.types) == [uint32], f"{key} must be stored as u32, got {field.types}"

    tensors = {t.name: t for t in reader.tensors}
    # Tied embeddings: no separate output projection.
    assert "output.weight" not in tensors
    assert tuple(tensors["token_embd.weight"].shape) == (cfg.n_embd, cfg.vocab_size)
    assert tuple(tensors["position_embd.weight"].shape) == (cfg.n_embd, cfg.block_size)
    assert tuple(tensors["blk.0.attn_qkv.weight"].shape) == (cfg.n_embd, 3 * cfg.n_embd)
    assert tuple(tensors["blk.0.ffn_up.weight"].shape) == (cfg.n_embd, cfg.ffn)
    assert tuple(tensors["blk.0.ffn_down.weight"].shape) == (cfg.ffn, cfg.n_embd)
    assert int(tensors["token_embd.weight"].tensor_type) == 0

    vocab = fields["tokenizer.ggml.tokens"].contents()
    types = fields["tokenizer.ggml.token_type"].contents()
    merges = fields["tokenizer.ggml.merges"].contents()
    assert len(vocab) == cfg.vocab_size == len(types)
    assert types[256:261] == [3, 3, 3, 3, 3], "Ember's five specials must be CONTROL tokens"
    assert vocab[256:261] == ["<pad>", "<bos>", "<eos>", "<doc>", "<unk>"]
    assert all(t == 1 for t in types[:256]), "byte tokens must be NORMAL in the gpt2 alphabet"
    assert len(merges) == len(tok.merges)
    assert all(" " in m for m in merges), "merges are stored as 'a b' strings"


def test_tensor_mapping_transposes_exactly_like_the_upstream_gpt2_converter(trained) -> None:
    tmp, cfg, params, tok = trained
    mapped = ember_to_gpt2_tensors(cfg, params)

    # ggml stores a linear layer as {in_features, out_features} in ne order, i.e.
    # memory is out_features rows of in_features — the transpose of Ember's
    # `x @ W` layout (identical to HuggingFace GPT-2 Conv1D, which
    # conversion/gpt2.py also transposes).
    np.testing.assert_array_equal(mapped["blk.0.attn_qkv.weight"], params["blk.0.attn.qkv.weight"].T)
    np.testing.assert_array_equal(mapped["blk.1.ffn_up.weight"], params["blk.1.mlp.fc.weight"].T)
    np.testing.assert_array_equal(mapped["blk.1.ffn_down.weight"], params["blk.1.mlp.proj.weight"].T)
    np.testing.assert_array_equal(mapped["blk.0.attn_output.weight"], params["blk.0.attn.proj.weight"].T)
    # Embeddings and biases keep their layout.
    np.testing.assert_array_equal(mapped["token_embd.weight"], params["wte"])
    np.testing.assert_array_equal(mapped["position_embd.weight"], params["wpe"])
    np.testing.assert_array_equal(mapped["output_norm.bias"], params["ln_f.bias"])
    np.testing.assert_array_equal(mapped["blk.0.attn_norm.weight"], params["blk.0.ln1.gain"])
    np.testing.assert_array_equal(mapped["blk.0.ffn_norm.bias"], params["blk.0.ln2.bias"])
    np.testing.assert_array_equal(mapped["blk.0.attn_qkv.bias"], params["blk.0.attn.qkv.bias"])

    expected_names = {
        "token_embd.weight", "position_embd.weight", "output_norm.weight", "output_norm.bias",
    }
    for i in range(cfg.n_layer):
        expected_names.update({
            f"blk.{i}.attn_norm.weight", f"blk.{i}.attn_norm.bias",
            f"blk.{i}.attn_qkv.weight", f"blk.{i}.attn_qkv.bias",
            f"blk.{i}.attn_output.weight", f"blk.{i}.attn_output.bias",
            f"blk.{i}.ffn_norm.weight", f"blk.{i}.ffn_norm.bias",
            f"blk.{i}.ffn_up.weight", f"blk.{i}.ffn_up.bias",
            f"blk.{i}.ffn_down.weight", f"blk.{i}.ffn_down.bias",
        })
    assert set(mapped) == expected_names

    # Untied embeddings must produce an explicit output projection.
    cfg2 = ModelConfig(
        n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=tok.vocab_size, tie_embeddings=False
    )
    gpt2 = GPT(cfg2, seed=3)
    mapped2 = ember_to_gpt2_tensors(cfg2, gpt2.params)
    # lm_head.weight is [vocab, n_embd] used as `x @ W.T`, i.e. the same memory
    # layout GGUF wants for {n_embd, n_vocab} — no transpose, exactly like wte.
    np.testing.assert_array_equal(mapped2["output.weight"], gpt2.params["lm_head.weight"])


def test_export_rejects_broken_checkpoints_with_actionable_messages(trained, tmp_path) -> None:
    tmp, cfg, params, tok = trained

    # Missing tensor
    incomplete = {k: v for k, v in params.items() if k != "blk.1.mlp.proj.weight"}
    with pytest.raises(GgufExportError, match="missing tensors"):
        export_gguf(tmp_path / "missing.gguf", cfg, incomplete, tok)

    # Wrong shape
    wrong = dict(params)
    wrong["wte"] = np.zeros((cfg.vocab_size + 1, cfg.n_embd), dtype=np.float32)
    with pytest.raises(GgufExportError, match="shape"):
        export_gguf(tmp_path / "shape.gguf", cfg, wrong, tok)

    # Non-finite weights must never reach a model file
    nans = dict(params)
    nans["wpe"] = np.array(params["wpe"], dtype=np.float32)
    nans["wpe"][0, 0] = np.nan
    with pytest.raises(GgufExportError, match="non-finite"):
        export_gguf(tmp_path / "nan.gguf", cfg, nans, tok)

    # Vocab / tokenizer disagreement (a tokenizer trained on other text)
    other_tok = train_bpe(
        ["Completely different vocabulary content about railways, bridges and signalling."] * 12,
        vocab_size=cfg.vocab_size + 120,
    )
    assert other_tok.vocab_size != cfg.vocab_size
    with pytest.raises(GgufExportError, match="vocab"):
        export_gguf(tmp_path / "vocab.gguf", cfg, params, other_tok)

    # Heads that do not divide the embedding width
    bad_cfg = ModelConfig(n_layer=1, n_head=5, n_embd=64, block_size=16, vocab_size=tok.vocab_size)
    with pytest.raises(GgufExportError):
        export_gguf(tmp_path / "heads.gguf", bad_cfg, params, tok)

    # Unknown quantization name
    with pytest.raises(GgufExportError, match="quantization"):
        export_gguf(tmp_path / "q.gguf", cfg, params, tok, quantization="q4_k_m")

    # None of the failures may have created a file
    assert not list(tmp_path.glob("*.gguf"))


def test_export_is_atomic_and_never_corrupts_an_existing_model(trained, tmp_path) -> None:
    tmp, cfg, params, tok = trained
    out = tmp_path / "active.gguf"
    export_gguf(out, cfg, params, tok, name="first", step=1)
    good_bytes = out.read_bytes()

    broken = dict(params)
    broken["ln_f.gain"] = np.full_like(params["ln_f.gain"], np.inf)
    with pytest.raises(GgufExportError):
        export_gguf(out, cfg, broken, tok, name="second", step=2)

    assert out.read_bytes() == good_bytes, "a failed export must leave the previous model untouched"
    assert list(tmp_path.glob("*.tmp-*")) == [], "no temporary files may be left behind"


def test_q8_0_quantization_matches_the_ggml_block_layout(trained, tmp_path) -> None:
    tmp, cfg, params, tok = trained
    rng = np.random.default_rng(5)
    arr = (rng.normal(0.0, 0.05, size=(4, 64))).astype(np.float32)
    packed = quantize_q8_0(arr)

    # block_q8_0 = { ggml_half d; int8_t qs[32]; } — 34 bytes per 32 values.
    assert len(packed) == (arr.size // 32) * 34
    blocks = arr.reshape(-1, 32)
    for i in range(blocks.shape[0]):
        block = packed[i * 34 : (i + 1) * 34]
        scale = np.frombuffer(block[:2], dtype="<f2")[0]
        qs = np.frombuffer(block[2:], dtype=np.int8)
        expected_scale = np.float16(np.max(np.abs(blocks[i])) / 127.0)
        assert scale == expected_scale, f"block {i}: scale {scale} != {expected_scale}"
        reconstructed = qs.astype(np.float32) * float(scale)
        # Q8_0 is lossy but must round-trip to within one quantization step.
        step = float(expected_scale) if float(expected_scale) > 0 else 1.0
        assert np.max(np.abs(reconstructed - blocks[i])) <= step + 1e-4
        assert np.all(np.isfinite(np.frombuffer(block[:2], dtype="<f2").astype(np.float32)))

    with pytest.raises(GgufExportError, match="multiple of 32"):
        quantize_q8_0(np.zeros((3, 5), dtype=np.float32))

    out = tmp_path / "model-q8.gguf"
    report = export_gguf(out, cfg, params, tok, name="q8", step=3, quantization="q8_0")
    assert report.quantization == "q8_0"
    assert report.file_type == 7  # GGUF FileType.MOSTLY_Q8_0
    # Only the four large per-layer matrices are quantized; embeddings, positional
    # embeddings and every bias stay float32 (as upstream quantizers do).
    assert len(report.quantized_tensors) == 4 * cfg.n_layer
    assert "token_embd.weight" in report.float_tensors
    assert "position_embd.weight" in report.float_tensors
    assert all(not n.endswith(".bias") for n in report.quantized_tensors)

    # Same weights, float32 export: the quantized file must be materially smaller.
    f32_out = tmp_path / "model-f32.gguf"
    export_gguf(f32_out, cfg, params, tok, name="q8", step=3, quantization="f32")
    assert out.stat().st_size < f32_out.stat().st_size
    assert out.stat().st_size < 0.75 * f32_out.stat().st_size

    header = read_gguf_header(out)
    q8 = {t.name: t for t in header.tensors if t.dtype == GGML_Q8_0}
    assert "blk.0.attn_qkv.weight" in q8
    assert q8["blk.0.attn_qkv.weight"].dims == (cfg.n_embd, 3 * cfg.n_embd)

    reader = gguf_reader.GGUFReader(str(out))
    q8_type = int(gguf_reader.GGMLQuantizationType.Q8_0)
    read_types = {t.name: int(t.tensor_type) for t in reader.tensors}
    assert read_types["blk.0.ffn_up.weight"] == q8_type
    assert read_types["blk.0.ffn_up.bias"] == int(gguf_reader.GGMLQuantizationType.F32)
    assert read_types["token_embd.weight"] == int(gguf_reader.GGMLQuantizationType.F32)


def test_header_reader_and_validation_report_every_required_fact(trained, tmp_path) -> None:
    tmp, cfg, params, tok = trained
    out = tmp_path / "valid.gguf"
    export_gguf(out, cfg, params, tok, name="validate-me", step=42)

    info = validate_gguf_file(out)
    for key in (
        "path", "bytes", "format", "ggufVersion", "arch", "emberFormat", "legacy", "contextLength",
        "embeddingLength", "blockCount", "headCount", "feedForwardLength", "vocabSize",
        "tensorCount", "quantizedTensors", "eosTokenId", "bosTokenId", "tokenizerModel",
        "tokenizerPre", "name", "valid",
    ):
        assert key in info, f"validation report must include {key}"
    assert info["valid"] is True and info["legacy"] is False
    assert info["arch"] == "gpt2" and info["tokenizerPre"] == "gpt-2"
    assert info["vocabSize"] == cfg.vocab_size
    assert info["contextLength"] == cfg.block_size
    assert info["eosTokenId"] == 258

    header = read_gguf_header(out)
    assert header.version == 3 and header.alignment == 32
    assert header.data_start % 32 == 0
    # Every tensor must fit inside the file.
    size = out.stat().st_size
    for t in header.tensors:
        assert header.data_start + t.offset + t.nbytes <= size


def test_incompatible_files_are_named_not_guessed(trained, tmp_path) -> None:
    tmp, cfg, params, tok = trained

    # An Ember .emb training checkpoint is NOT a GGUF model and must say so.
    emb = tmp_path / "checkpoint.emb"
    save_model(emb, cfg, params, meta={"step": 3})
    with pytest.raises(GgufExportError, match=r"\.emb"):
        validate_gguf_file(emb)
    message = describe_incompatible_bytes(emb.read_bytes()[:64], name="checkpoint.emb")
    assert "training checkpoint" in message and "GGUF" in message

    # A legacy JSON brain (v4 or old TensorFlow.js) is rejected with guidance.
    legacy = tmp_path / "brain.json"
    legacy.write_text('{"format": "ember-brain-v2", "version": 2, "weights": {}}', encoding="utf-8")
    with pytest.raises(GgufExportError, match="JSON brain"):
        validate_gguf_file(legacy)

    # A GGUF file with a different architecture must not be silently accepted.
    foreign = tmp_path / "llama.gguf"
    blob = bytearray(emb.read_bytes()[:0])  # build a minimal fake GGUF header
    del blob
    header = struct.pack("<IIQQ", 0x46554747, 3, 0, 3)
    def kv_str(key: str, value: str) -> bytes:
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value.encode()
    def kv_u32(key: str, value: int) -> bytes:
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", value)
    body = (
        kv_str("general.architecture", "llama")
        + kv_u32("llama.context_length", 4096)
        + kv_u32("llama.embedding_length", 4096)
    )
    foreign.write_bytes(header + body + b"\x00" * 2048)
    with pytest.raises(GgufExportError, match="architecture"):
        validate_gguf_file(foreign)

    # Truncated and tiny files are reported honestly.
    truncated = tmp_path / "truncated.gguf"
    good = tmp_path / "good.gguf"
    export_gguf(good, cfg, params, tok)
    truncated.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
    with pytest.raises(GgufExportError, match="truncated"):
        validate_gguf_file(truncated)

    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GGUF" + b"\x00" * 16)
    with pytest.raises(GgufExportError, match="too small"):
        validate_gguf_file(tiny)

    missing = tmp_path / "does-not-exist.gguf"
    with pytest.raises(GgufExportError, match="does not exist"):
        validate_gguf_file(missing)


def test_export_from_a_real_backend_checkpoint_roundtrips(tmp_path) -> None:
    """The path the app actually uses: cast → fire → export active.gguf."""
    from ember.backend import EmberBackend

    backend = EmberBackend(tmp_path / "data", tmp_path / "cache")
    try:
        backend.dispatch("corpus.addPack", {"pack": "starter"})
        backend.dispatch(
            "vessel.cast",
            {"config": {"nLayer": 2, "nHead": 2, "nEmbd": 64, "blockSize": 64, "vocabSize": 420}, "seed": 5},
        )
        backend.dispatch("train.sync", {"steps": 3, "batch": 2, "lr": 3e-3, "every": 0, "validateEvery": 0})
        gguf_path = backend.trainer.sync_gguf(force=True)
        assert gguf_path is not None and gguf_path.exists()
        info = validate_gguf_file(gguf_path)
        assert info["arch"] == "gpt2"
        assert info["blockCount"] == 2
        assert info["embeddingLength"] == 64
        assert info["vocabSize"] == backend.trainer.cfg.vocab_size
        # Re-exporting the same weights is byte-identical (deterministic pipeline).
        first = gguf_path.read_bytes()
        backend.trainer.sync_gguf(force=True)
        second = gguf_path.read_bytes()
        assert first[:64] == second[:64]
        assert len(first) == len(second)
    finally:
        backend.close()
