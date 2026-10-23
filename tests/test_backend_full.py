"""Comprehensive backend tests: dataset, attention masks, gradients, AdamW,
validation non-mutation, checkpoints, legacy brain handling, all 35 ingestion
fixtures, pause/resume/cancel, Android bridge callbacks, and HTTP server RPC."""

from __future__ import annotations

import json
import struct
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ember.backend import EmberBackend, init_android, rpc_json, shutdown_android
from ember.corpus import Corpus
from ember.dataset import (
    IGNORE_INDEX,
    Batcher,
    Document,
    build_train_val_streams,
    split_documents,
)
from ember.ingest import ImportManager, inspect_bytes
from ember.model import FORMAT_MAGIC, GPT, ModelConfig, load_model_bytes, save_model
from ember.optim import AdamW, LRSchedule, clip_grad_norm
from ember.server import create_server
from ember.spec import SPECIAL_BOS, SPECIAL_EOS, SPECIAL_PAD
from ember.tokenizer import train_bpe

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _sample_docs(n: int = 10) -> list[Document]:
    out = []
    for i in range(n):
        out.append(
            Document(
                id=f"doc-{i}",
                title=f"Topic {i}",
                source="wiki" if i % 2 == 0 else "notes",
                kind="lesson" if i == n - 1 else "page",
                weight=6.0 if i == n - 1 else 1.0,
                text=(
                    f"### Prompt:\nExplain item {i}\n\n### Response:\nItem {i} demonstrates deterministic learning."
                    if i == n - 1
                    else f"Document number {i}: stars shine brightly across the clear night sky over harbour {i}. " * 4
                ),
            )
        )
    return out


def test_dataset_disjoint_split_and_lesson_prompt_masking() -> None:
    docs = _sample_docs(8)
    tok = train_bpe([d.text for d in docs], vocab_size=360)

    train_docs, val_docs = split_documents(docs, holdout_frac=0.15)
    assert len(train_docs) >= 1 and len(val_docs) >= 1
    assert {d.id for d in train_docs}.isdisjoint({d.id for d in val_docs})

    t_stream, v_stream = build_train_val_streams(
        docs,
        tok,
        holdout_frac=0.15,
        mask_lesson_prompts=True,
        balance="sqrt",
    )
    assert t_stream.n_tokens > 0 and v_stream.n_tokens > 0
    assert t_stream.ids[0] == SPECIAL_BOS
    assert t_stream.ids[-1] == SPECIAL_EOS
    # Lesson span must have masked prompt targets at its start
    lesson_spans = [s for s in t_stream.spans if s.kind == "lesson"]
    assert len(lesson_spans) == 1
    lsp = lesson_spans[0]
    lesson_targets = t_stream.targets[lsp.start : lsp.start + lsp.length]
    assert np.all(lesson_targets[:5] == IGNORE_INDEX), "Lesson prompt tokens must be masked with IGNORE_INDEX"
    assert np.any(lesson_targets[5:-1] != IGNORE_INDEX), "Lesson response tokens must retain valid loss targets"

    batcher = Batcher(t_stream, block_size=24, batch_size=4, seed=7, sampler="epoch", boundaries="respect")
    x, y, doc_ids = batcher.batch()
    assert x.shape == (4, 24)
    assert y.shape == (4, 24)
    assert doc_ids is not None and doc_ids.shape == (4, 24)
    # Any padded slots must have IGNORE_INDEX targets
    pad_mask = x == SPECIAL_PAD
    if np.any(pad_mask):
        assert np.all(y[pad_mask] == IGNORE_INDEX)


def test_document_boundary_attention_isolation() -> None:
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, block_size=12, vocab_size=300)
    gpt = GPT(cfg, seed=123)
    x1 = np.array([[257, 40, 41, 42, 258, 257, 70, 71, 72, 73, 74, 258]], dtype=np.int32)
    x2 = x1.copy()
    # Mutate all tokens inside the first document only
    x2[0, 1:4] = [99, 100, 101]
    doc_ids = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int32)

    logits1, _ = gpt.forward(x1, doc_ids=doc_ids)
    logits2, _ = gpt.forward(x2, doc_ids=doc_ids)
    # Document 1 positions (indices 5..11) must be 100% invariant to Document 0 tokens!
    np.testing.assert_allclose(logits1[0, 5:], logits2[0, 5:], atol=1e-6)


def test_analytic_gradients_match_numerical_finite_differences() -> None:
    rng = np.random.default_rng(42)
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, block_size=10, vocab_size=300)
    gpt = GPT(cfg, seed=9)
    x = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    doc = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2]], dtype=np.int32)
    y = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    y[0, :2] = IGNORE_INDEX

    _, _ = gpt.forward(x, doc_ids=doc, targets=y)
    grads = {k: v.copy() for k, v in gpt.backward().items()}

    for name in (
        "wte",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight",
        "output_norm.weight",
    ):
        arr = gpt.params[name].reshape(-1)
        g_arr = grads[name].reshape(-1)
        for idx in rng.choice(arr.size, size=4, replace=False):
            orig = float(arr[idx])
            eps = 2e-3
            arr[idx] = orig + eps
            _, lp = gpt.forward(x, doc_ids=doc, targets=y)
            arr[idx] = orig - eps
            _, lm = gpt.forward(x, doc_ids=doc, targets=y)
            arr[idx] = orig
            num = (float(lp) - float(lm)) / (2.0 * eps)
            ana = float(g_arr[idx])
            assert abs(num - ana) < 2e-3 + 0.04 * abs(ana), f"Grad mismatch on {name}[{idx}]: {num} vs {ana}"


def test_training_updates_weights_and_validation_never_mutates_weights() -> None:
    with tempfile.TemporaryDirectory() as td:
        be = EmberBackend(td)
        be.dispatch("corpus.addPack", {"pack": "starter"})
        be.dispatch("corpus.addPack", {"pack": "skills"})
        be.dispatch("vessel.cast", {"nLayer": 2, "nHead": 2, "nEmbd": 48, "blockSize": 32, "vocabSize": 500})

        before_weights = {k: v.copy() for k, v in be.trainer.model.params.items()}

        # Validation alone must NOT touch any parameter or optimizer moment
        v1 = be.trainer.evaluate_validation()
        assert v1 is not None and np.isfinite(v1["loss"])
        for k, v in be.trainer.model.params.items():
            assert np.array_equal(v, before_weights[k]), f"Validation mutated {k}!"
        assert be.trainer.optimizer.t == 0

        # Real training MUST update weights and reduce loss
        run = be.dispatch("train.sync", {"steps": 50, "batch": 4, "lr": 1.5e-3, "every": 25, "validateEvery": 25, "gradClip": 1.0})
        assert run["completedSteps"] == 50
        assert be.trainer.step == 50
        assert be.trainer.losses[-1] < be.trainer.losses[0]
        changed = sum(1 for k, v in be.trainer.model.params.items() if not np.array_equal(v, before_weights[k]))
        assert changed == len(before_weights), "Every parameter tensor in the model must receive real updates"

        # Checkpoint save + corrupt check + reload
        ckpt = be.dispatch("ckpt.save", {"name": "milestone-50"})
        assert ckpt["step"] == 50
        saved_wte = be.trainer.model.params["wte"].copy()
        be.trainer.model.params["wte"].fill(0.0)
        be.dispatch("ckpt.load", {"name": "milestone-50"})
        np.testing.assert_array_equal(be.trainer.model.params["wte"], saved_wte)

        # Portable v4 brain export/import round-trip
        brain = be.dispatch("brain.exportJson", {"name": "portable-v4", "includeCorpus": True})
        assert brain["version"] == 4
        with tempfile.TemporaryDirectory() as td2:
            be2 = EmberBackend(td2)
            imp = be2.dispatch("brain.importText", {"text": json.dumps(brain)})
            assert imp["imported"] is True
            assert imp["step"] == 50
            np.testing.assert_allclose(be2.trainer.model.params["wte"], saved_wte, atol=0.0)
            be2.close()

        # Legacy JS brain must be explicitly rejected while preserving embedded corpus docs
        legacy_js_brain = {
            "format": "ember-brain-v2",
            "version": 2,
            "cfg": {"nLayer": 2, "nHead": 2, "nEmbd": 64, "blockSize": 64, "vocabSize": 500},
            "weights": {},
            "corpusDocs": [{"title": "Recovered Legacy Doc", "text": "Useful user writing recovered from old JS brain export."}],
        }
        with pytest.raises(ValueError, match="Incompatible legacy brain format"):
            be.dispatch("brain.importText", {"text": json.dumps(legacy_js_brain)})
        assert any("Recovered Legacy Doc" in d["title"] for d in be.dispatch("corpus.list", {"limit": 50})["items"])
        be.close()


def test_corrupt_emb_checksum_and_header_rejection() -> None:
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=300)
    gpt = GPT(cfg, seed=1)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.emb"
        save_model(p, cfg, gpt.params)
        raw = bytearray(p.read_bytes())
        # Flip one bit inside float payload -> CRC32 must catch corruption!
        raw[-20] ^= 0xFF
        with pytest.raises(ValueError, match="checksum"):
            load_model_bytes(bytes(raw))

        # Wrong magic -> clean error
        with pytest.raises(ValueError, match="old JavaScript engine"):
            load_model_bytes(b'{"format": "ember-brain-v1"}')

        # Truncated file
        with pytest.raises(ValueError):
            load_model_bytes(FORMAT_MAGIC + struct.pack("<II", 3, 9999))


def test_all_35_ingestion_fixtures_and_pause_resume_lifecycle() -> None:
    expected_bad = {
        "bomb.zip",
        "corrupt.jsonl.gz",
        "corrupt.zip",
        "traversal.zip",
        "truncated.jsonl.bz2",
        "truncated.jsonl.gz",
        "truncated.jsonl.xz",
        "truncated.jsonl.zst",
        "truncated.parquet",
    }
    for f in sorted(FIXTURES_DIR.iterdir()):
        if f.suffix == ".py":
            continue
        if f.name in expected_bad:
            with pytest.raises(ValueError):
                inspect_bytes(f.read_bytes(), name=f.name)
        else:
            prev = inspect_bytes(f.read_bytes(), name=f.name)
            if f.name == "numbers_only.parquet":
                assert len(prev["documents"]) == 0
            else:
                assert len(prev["documents"]) >= 1, f"Expected documents from {f.name}"

    # Pause + Resume durability test on multi-hundred-record JSONL
    with tempfile.TemporaryDirectory() as td:
        big_jsonl = Path(td) / "many.jsonl"
        big_jsonl.write_text(
            "\n".join(json.dumps({"title": f"Row {i}", "text": f"Unique dataset passage number {i} for streaming import test."}) for i in range(250)),
            encoding="utf-8",
        )
        corpus = Corpus(Path(td) / "db")
        mgr = ImportManager(corpus)
        mgr.stage_path("tok1", big_jsonl, name="many.jsonl")
        # Trigger pause after first batch commits
        paused_once = False

        def _prog(ev: dict) -> None:
            nonlocal paused_once
            if not paused_once and ev["records"] >= 100:
                paused_once = True
                mgr.pause("tok1")

        r1 = mgr.commit("tok1", on_progress=_prog)
        assert r1["outcome"] == "paused"
        assert corpus.count() == 100
        assert len(mgr.list_imports()) == 1

        r2 = mgr.resume("tok1")
        assert r2["outcome"] == "completed"
        assert corpus.count() == 250
        corpus.close()


def test_background_cancellation_android_bridge_and_http_server() -> None:
    events: list[tuple[str, dict]] = []
    synced: list[tuple[str, str]] = []

    class MockKotlinBridge:
        def onPythonEvent(self, name: str, payload: str) -> None:
            events.append((name, json.loads(payload)))

        def onModelSynced(self, mpath: str, tpath: str) -> None:
            synced.append((mpath, tpath))

    with tempfile.TemporaryDirectory() as td:
        init_res = json.loads(init_android(str(Path(td) / "data"), str(Path(td) / "cache"), MockKotlinBridge()))
        assert init_res["ok"] is True
        try:
            r = json.loads(rpc_json(json.dumps({"method": "corpus.addPack", "params": {"pack": "starter"}})))
            assert r["ok"] is True and r["result"]["added"] >= 5

            cast = json.loads(rpc_json(json.dumps({"method": "vessel.cast", "params": {"preset": "phone"}})))
            assert cast["ok"] is True and cast["result"]["ready"] is True
            assert len(synced) >= 1 and Path(synced[-1][0]).exists()

            # Start background training and test live cancellation
            started = json.loads(rpc_json(json.dumps({"method": "train.start", "params": {"steps": 500, "batch": 2}})))
            assert started["ok"] is True
            time.sleep(0.15)
            rpc_json(json.dumps({"method": "train.stop", "params": {}}))
            deadline = time.time() + 10.0
            while time.time() < deadline:
                st = json.loads(rpc_json(json.dumps({"method": "train.status", "params": {}})))["result"]
                if not st["training"]["running"]:
                    break
                time.sleep(0.05)
            assert any(name in ("train.progress", "train.done") for name, _ in events)
        finally:
            shutdown_android()

    # Live HTTP server `/api/rpc` smoke test
    with tempfile.TemporaryDirectory() as td_srv:
        srv = create_server(host="127.0.0.1", port=0, data_dir=td_srv)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/rpc",
                data=json.dumps({"method": "ping", "params": {}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assert body["ok"] is True and body["result"]["pong"] is True
        finally:
            srv.shutdown()
            srv.server_close()
