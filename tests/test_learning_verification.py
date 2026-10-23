"""Controlled verification that the Ember v3 training pipeline really learns.

A training loop that runs to completion proves very little on its own, so these
tests measure the mathematics and the mechanics on a small deterministic
fixture:

* the forward pass returns the documented shapes and a finite loss,
* gradients exist, are finite, and match the parameters they belong to,
* the optimizer changes *every* trainable tensor,
* loss falls and next-token accuracy rises on a fixed, held-fixed batch,
* validation never mutates weights or optimizer state,
* checkpoints restore weights *and* Adam moments bit-for-bit,
* the restored model produces identical logits in Python and in compiled C++,
* a diverging run is refused instead of writing NaN weights to `active.emb`,
* cancellation stops promptly and keeps the completed weights,
* repeated generation requests stay deterministic (no leaked KV/PRNG state).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from ember.backend import EmberBackend
from ember.dataset import IGNORE_INDEX, Batcher
from ember.inference import LlamaCppEngine, EngineConfig, GenParams, InferenceError

# Eight short documents with a strongly learnable, repetitive structure. Small on
# purpose: the point is to verify the implementation, not to train a good model.
LEARNABLE_DOCS = [
    ("Tides", "The tide rises and the tide falls. The tide rises and the tide falls."),
    ("Stars", "Stars rise in the east and set in the west. Stars rise in the east."),
    ("Bread", "Flour and water and salt make bread. Flour and water and salt make bread."),
    ("Rain", "Clouds gather, clouds darken, clouds release the rain. Clouds gather."),
    ("Reefs", "Coral reefs shelter fish from the open ocean. Coral reefs shelter fish."),
    ("Trains", "The train leaves the station and crosses the bridge. The train leaves."),
    ("Wind", "Wind bends the palms and pushes the sails. Wind bends the palms."),
    ("Rivers", "The river carries silt to the delta and the sea. The river carries silt."),
]

TINY = {
    "config": {"nLayer": 2, "nHead": 2, "nEmbd": 32, "blockSize": 24, "vocabSize": 400, "dropout": 0.0},
    "lr": 3e-3,
    # Deterministic initialization: without a seed every cast draws fresh random
    # weights and the measured improvement varies run to run.
    "seed": 20261023,
}


@pytest.fixture()
def learnable_backend(tmp_path):
    be = EmberBackend(tmp_path / "data", tmp_path / "cache")
    for title, text in LEARNABLE_DOCS:
        be.dispatch("corpus.addText", {"title": title, "text": text, "weight": 3})
    be.dispatch("vessel.cast", TINY)
    yield be
    be.close()


def _fixed_batch(be: EmberBackend, seed: int = 1234):
    trainer = be.trainer
    batcher = Batcher(
        trainer.train_stream,
        block_size=trainer.cfg.block_size,
        batch_size=4,
        seed=seed,
        sampler="random",
        boundaries="respect",
    )
    return batcher.batch()


def _evaluate(be: EmberBackend, x, y, doc_ids) -> tuple[float, float]:
    """Return (mean loss, top-1 next-token accuracy) without leaving caches behind."""
    model = be.trainer.model
    logits, loss = model.forward(x, doc_ids=doc_ids, targets=y, dropout=0.0)
    model._cache = None
    assert logits.shape == (x.shape[0], x.shape[1], be.trainer.cfg.vocab_size)
    valid = y != IGNORE_INDEX
    assert valid.any(), "the evaluation batch must contain graded tokens"
    preds = np.argmax(logits, axis=-1)
    acc = float((preds[valid] == y[valid]).mean())
    assert loss is not None and np.isfinite(loss), f"non-finite loss: {loss}"
    return float(loss), acc


def test_forward_shapes_and_finite_loss_before_any_training(learnable_backend) -> None:
    be = learnable_backend
    x, y, doc_ids = _fixed_batch(be)
    assert x.shape == (4, be.trainer.cfg.block_size)
    assert y.shape == x.shape
    assert doc_ids is not None and doc_ids.shape == x.shape
    loss, acc = _evaluate(be, x, y, doc_ids)
    vocab = be.trainer.cfg.vocab_size
    # An untrained model sits near uniform: loss ≈ ln(vocab), accuracy ≈ 1/vocab.
    assert 0.5 * np.log(vocab) < loss < 2.0 * np.log(vocab), f"unexpected initial loss {loss}"
    assert acc < 0.25, f"an untrained model should not already predict this corpus (acc={acc})"


def test_gradients_are_finite_and_the_optimizer_updates_every_tensor(learnable_backend) -> None:
    be = learnable_backend
    trainer = be.trainer
    x, y, doc_ids = _fixed_batch(be)

    before = {k: v.copy() for k, v in trainer.model.params.items()}
    trainer.model.zero_grad()
    _, loss = trainer.model.forward(x, doc_ids=doc_ids, targets=y, dropout=0.0)
    grads = trainer.model.backward()
    trainer.model._cache = None

    assert np.isfinite(loss)
    assert set(grads) == set(trainer.model.params), "every parameter must receive a gradient"
    for name, g in grads.items():
        assert g.shape == trainer.model.params[name].shape, f"gradient shape mismatch for {name}"
        assert np.all(np.isfinite(g)), f"non-finite gradient in {name}"
    nonzero = [n for n, g in grads.items() if np.any(g != 0.0)]
    assert len(nonzero) >= len(grads) - 1, "almost every gradient should be non-zero"

    trainer.optimizer.set_lr(3e-3)
    trainer.optimizer.step(grads)
    changed = [k for k in before if not np.array_equal(before[k], trainer.model.params[k])]
    assert len(changed) == len(before), f"optimizer left tensors untouched: {set(before) - set(changed)}"
    assert trainer.optimizer.t == 1


def test_training_reduces_loss_and_raises_next_token_accuracy(learnable_backend) -> None:
    be = learnable_backend
    x, y, doc_ids = _fixed_batch(be)
    loss_before, acc_before = _evaluate(be, x, y, doc_ids)

    run = be.dispatch(
        "train.sync",
        {"steps": 120, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 60, "gradClip": 1.0},
    )
    assert run["completedSteps"] == 120
    assert run["outcome"] == "completed"

    loss_after, acc_after = _evaluate(be, x, y, doc_ids)
    # Measured deterministically on this fixture (fixed init seed + content-derived
    # document ids): loss 5.8711 -> 1.8248, top-1 accuracy 0.000 -> 0.720.
    assert loss_after < loss_before * 0.5, f"loss barely moved: {loss_before:.4f} -> {loss_after:.4f}"
    assert loss_after < loss_before - 2.0, f"loss barely moved: {loss_before:.4f} -> {loss_after:.4f}"
    assert acc_after > acc_before + 0.4, f"accuracy did not improve: {acc_before:.3f} -> {acc_after:.3f}"

    losses = be.trainer.losses
    assert len(losses) == 120 and all(np.isfinite(l) for l in losses)
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) * 0.6, "the loss trend must fall across the run"

    # Held-out validation must improve too — otherwise the model only memorized
    # the training windows.
    validation = be.trainer.validations
    assert len(validation) == 2
    assert all(np.isfinite(v["loss"]) and v["ppl"] > 0 for v in validation)
    assert validation[-1]["loss"] < validation[0]["loss"], (
        f"validation loss did not improve: {validation[0]['loss']:.4f} -> {validation[-1]['loss']:.4f}"
    )
    assert validation[-1]["bestLoss"] <= validation[0]["loss"]


def test_validation_never_mutates_weights_or_optimizer_state(learnable_backend) -> None:
    be = learnable_backend
    be.dispatch("train.sync", {"steps": 5, "batch": 2, "lr": 3e-3, "every": 0, "validateEvery": 0})

    params_before = {k: v.copy() for k, v in be.trainer.model.params.items()}
    moments_before = {k: v.copy() for k, v in be.trainer.optimizer.m.items()}
    second_before = {k: v.copy() for k, v in be.trainer.optimizer.v.items()}
    t_before = be.trainer.optimizer.t

    point = be.trainer.evaluate_validation()
    assert point is not None and np.isfinite(point["loss"])

    for k, v in be.trainer.model.params.items():
        assert np.array_equal(v, params_before[k]), f"validation mutated parameter {k}"
    for k, v in be.trainer.optimizer.m.items():
        assert np.array_equal(v, moments_before[k]), f"validation mutated Adam moment m[{k}]"
    for k, v in be.trainer.optimizer.v.items():
        assert np.array_equal(v, second_before[k]), f"validation mutated Adam moment v[{k}]"
    assert be.trainer.optimizer.t == t_before, "validation must not advance the optimizer step counter"


def test_checkpoint_restores_weights_and_optimizer_state_bit_for_bit(learnable_backend, tmp_path) -> None:
    be = learnable_backend
    be.dispatch("train.sync", {"steps": 20, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
    saved = be.dispatch("ckpt.save", {"name": "learning-check"})
    assert saved["step"] == be.trainer.step

    params_at_save = {k: v.copy() for k, v in be.trainer.model.params.items()}
    moments_at_save = {k: v.copy() for k, v in be.trainer.optimizer.m.items()}
    second_at_save = {k: v.copy() for k, v in be.trainer.optimizer.v.items()}
    step_at_save = be.trainer.step

    # Keep training so the live model diverges from the checkpoint.
    be.dispatch("train.sync", {"steps": 10, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
    assert be.trainer.step == step_at_save + 10
    assert not np.array_equal(be.trainer.model.params["wte"], params_at_save["wte"])

    be.dispatch("ckpt.load", {"name": "learning-check"})
    assert be.trainer.step == step_at_save
    for k, v in params_at_save.items():
        np.testing.assert_array_equal(be.trainer.model.params[k], v, err_msg=f"weights differ for {k}")
    assert be.trainer.optimizer.t > 0, "the Adam step counter must be restored"
    for k, v in moments_at_save.items():
        np.testing.assert_array_equal(be.trainer.optimizer.m[k], v, err_msg=f"Adam m differs for {k}")
    for k, v in second_at_save.items():
        np.testing.assert_array_equal(be.trainer.optimizer.v[k], v, err_msg=f"Adam v differs for {k}")

    # Restoring must also repair the on-disk model used by native inference.
    assert be.trainer.active_model_path.exists()
    assert be.trainer.active_optimizer_path.exists()


def test_reloaded_model_matches_python_logits_and_greedy_tokens_in_cpp(learnable_backend) -> None:
    be = learnable_backend
    be.dispatch("train.sync", {"steps": 30, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
    path = be.trainer.sync_gguf(force=True)
    engine = LlamaCppEngine()
    assert engine.available(), engine.load_error
    try:
        engine.load_model(path, EngineConfig(n_ctx=64, n_batch=64, n_threads=2))
        ids = [257] + be.trainer.tokenizer.encode("The tide rises")
        expected = be.trainer.model.forward_logits(np.asarray([ids], dtype=np.int32))[0, -1]
        np.testing.assert_allclose(engine.forward_logits(ids), expected, atol=2e-4, rtol=2e-4)
        options = GenParams(max_tokens=8, temperature=0, top_k=1, repeat_penalty=1, seed=99, stop_on_eos=False)
        a = engine.generate("The tide", options)
        b = engine.generate("The tide", options)
        assert a.token_ids == b.token_ids and a.text == b.text
        assert a.new_tokens == 8
    finally:
        engine.close()


def test_diverging_run_is_refused_instead_of_writing_nan_weights(tmp_path) -> None:
    """An absurd learning rate must trip the finite-loss / finite-gradient guards.

    The requirement is not that training survives a ridiculous LR — it is that the
    guards refuse the update, the weights stay finite, and the corrupted state is
    never handed to native inference through `active.emb`.
    """
    be = EmberBackend(tmp_path / "data", tmp_path / "cache")
    try:
        for title, text in LEARNABLE_DOCS:
            be.dispatch("corpus.addText", {"title": title, "text": text, "weight": 2})
        be.dispatch("vessel.cast", TINY)

        run = be.dispatch(
            "train.sync",
            {"steps": 25, "batch": 4, "lr": 500.0, "every": 0, "validateEvery": 0, "gradClip": 1.0},
        )
        assert run["outcome"] in {"completed", "failed"}

        for name, arr in be.trainer.model.params.items():
            assert np.all(np.isfinite(arr)), f"parameter {name} became non-finite during a diverging run"
        recorded = [l for l in be.trainer.losses if np.isfinite(l)]
        assert len(recorded) == len(be.trainer.losses), "non-finite losses must never be recorded as metrics"

        # Native loading must succeed for finite tensor bytes, but an overflowed
        # graph must never reach the sampler unchecked: require finite logits or
        # an explicit structured decode failure, not a native crash.
        engine = LlamaCppEngine()
        assert engine.available(), engine.load_error
        try:
            engine.load_model(be.trainer.sync_gguf(force=True), EngineConfig(n_ctx=64, n_threads=2))
            try:
                logits = engine.forward_logits([257])
            except InferenceError as exc:
                assert exc.code == -9 and "non-finite" in str(exc)
            else:
                assert np.isfinite(logits).all()
        finally:
            engine.close()
    finally:
        be.close()


def test_cancellation_stops_promptly_and_keeps_completed_weights(learnable_backend) -> None:
    be = learnable_backend
    be.dispatch("train.start", {"steps": 5000, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
    deadline = time.time() + 30
    while time.time() < deadline and be.trainer.step < 2:
        time.sleep(0.05)
    assert be.trainer.step >= 1, "training never got going"

    t0 = time.perf_counter()
    be.dispatch("train.stop")
    deadline = time.time() + 60
    while time.time() < deadline and (be.trainer.running or be.trainer.starting):
        time.sleep(0.05)
    stop_seconds = time.perf_counter() - t0

    assert be.trainer.running is False
    assert be.trainer.outcome == "doused"
    assert be.trainer.step < 5000, "cancellation must actually stop the run"
    assert stop_seconds < 30, f"cancellation took {stop_seconds:.1f}s"
    for name, arr in be.trainer.model.params.items():
        assert np.all(np.isfinite(arr)), f"{name} is non-finite after cancellation"
    assert (be.trainer.ckpts_dir / "autosave.emb").exists(), "completed work must be checkpointed on stop"


def test_douse_requested_immediately_after_fire_is_not_lost(learnable_backend) -> None:
    """Regression: `train_sync` cleared the stop event on entry, so a Douse issued
    between launching the worker thread and reaching that line was discarded and
    the run trained to completion anyway."""
    be = learnable_backend
    started = be.dispatch(
        "train.start", {"steps": 2000, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0}
    )
    assert started["started"] is True

    # Stop immediately: the worker thread has not necessarily reached train_sync.
    be.dispatch("train.stop")

    deadline = time.time() + 60
    while time.time() < deadline and (be.trainer.running or be.trainer.starting):
        time.sleep(0.02)

    assert be.trainer.running is False and be.trainer.starting is False, "training never stopped"
    assert be.trainer.outcome == "doused", f"the stop request was lost (outcome={be.trainer.outcome})"
    assert be.trainer.step < 2000, "the run trained to completion despite being doused"


def test_repeated_generation_requests_stay_deterministic_and_finite(learnable_backend) -> None:
    be = learnable_backend
    be.dispatch("train.sync", {"steps": 20, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})

    greedy = {
        "prompt": "Stars rise in the east",
        "maxNew": 24,
        "temperature": 0.0,
        "topK": 1,
        "seed": "777",
        "stopOnBoundary": False,
    }
    first = be.dispatch("gen.run", dict(greedy))
    for i in range(11):
        again = be.dispatch("gen.run", dict(greedy))
        assert again["text"] == first["text"], f"greedy generation drifted on request {i + 2}"
        assert again["newTokens"] == 24
        assert again["ms"] > 0

    # Sampling with a fixed seed must also repeat exactly, and a different seed
    # must be free to differ (i.e. the PRNG is really seeded per request).
    sampled = {**greedy, "temperature": 0.9, "topK": 40, "topP": 0.95, "repetition": 1.15, "seed": "31337"}
    a = be.dispatch("gen.run", dict(sampled))
    b = be.dispatch("gen.run", dict(sampled))
    assert a["text"] == b["text"], "seeded sampling is not reproducible"

    # Oversized prompts must be truncated with a warning, not rejected or crashed.
    long_prompt = "the tide rises and falls " * 40
    over = be.dispatch("gen.run", {**greedy, "prompt": long_prompt, "maxNew": 8})
    assert over["warning"], "an over-length prompt must report the context-window truncation"
    assert over["newTokens"] == 8

    # A zero-token request is a valid no-op rather than an error.
    zero = be.dispatch("gen.run", {**greedy, "maxNew": 0})
    assert zero["newTokens"] == 0 and zero["text"] == ""


def test_cast_is_reproducible_with_an_explicit_seed(tmp_path) -> None:
    """Regression: random UUID document ids made the train/validation holdout split
    (a hash of the document id) change on every import, so two casts over identical
    text produced different streams — and different results — even with a fixed
    model seed. Ids are now derived from the content hash."""
    def build(root: Path) -> tuple[list[float], list[str], list[str]]:
        be = EmberBackend(root / "data", root / "cache")
        try:
            for title, text in LEARNABLE_DOCS:
                be.dispatch("corpus.addText", {"title": title, "text": text, "weight": 3})
            be.dispatch("vessel.cast", TINY)
            ids = [d["id"] for d in be.dispatch("corpus.list", {"limit": 50})["items"]]
            be.dispatch("train.sync", {"steps": 12, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
            return list(be.trainer.losses), sorted(ids), [l["line"] for l in be.trainer.train_log if "Cast" in l["line"]]
        finally:
            be.close()

    losses_a, ids_a, cast_a = build(tmp_path / "a")
    losses_b, ids_b, cast_b = build(tmp_path / "b")

    assert ids_a == ids_b, "document ids must be a deterministic function of content"
    assert cast_a == cast_b, "the cast log (train/val token counts) must match"
    assert losses_a == pytest.approx(losses_b, abs=0.0, rel=0.0), "identical corpus + seed must train identically"
    assert len(losses_a) == 12


def test_active_model_survives_a_backend_restart(learnable_backend, tmp_path) -> None:
    """The app must come back with the same brain after the process is recreated."""
    be = learnable_backend
    be.dispatch("train.sync", {"steps": 12, "batch": 4, "lr": 3e-3, "every": 0, "validateEvery": 0})
    step_before = be.trainer.step
    params_before = {k: v.copy() for k, v in be.trainer.model.params.items()}
    data_dir = be.data_dir
    cache_dir = be.cache_dir
    be.close()

    restarted = EmberBackend(data_dir, cache_dir)
    try:
        assert restarted.trainer.ready, "active.emb + active.tok.json must restore the vessel on startup"
        assert restarted.trainer.step == step_before
        for k, v in params_before.items():
            np.testing.assert_array_equal(restarted.trainer.model.params[k], v, err_msg=f"{k} changed across restart")
        status = restarted.dispatch("status")
        assert status["model"]["ready"] is True
        assert status["corpus"]["documents"] == len(LEARNABLE_DOCS), "the corpus must survive the restart"
        gen = restarted.dispatch("gen.run", {"prompt": "The tide", "maxNew": 8, "temperature": 0.0, "topK": 1, "seed": "3"})
        assert gen["newTokens"] == 8
    finally:
        restarted.close()
