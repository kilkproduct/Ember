"""Real Python training controller, checkpoint manager, and brain serializer."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corpus import Corpus
from .dataset import (
    IGNORE_INDEX,
    Batcher,
    TokenStream,
    build_train_val_streams,
    compute_holdout_ids,
)
from .model import (
    BRAIN_KIND,
    BRAIN_VERSION,
    GPT,
    ModelConfig,
    check_params,
    load_model,
    load_model_bytes,
    model_from_dict,
    model_to_dict,
    save_model,
)
from .optim import AdamW, LRSchedule, clip_grad_norm, grad_has_nan
from .sampler import (
    GenerationResult,
    NativeInferenceBridge,
    reference_generate,
)
from .spec import (
    PRESETS,
    TOKENIZER_TYPE,
    estimate_params,
    estimate_train_bytes,
)
from .tokenizer import BPETokenizer, train_bpe

# Practical phone training ceilings (bytes of parameters + moments + activations)
WARN_PHONE_BYTES = 160 * 1024 * 1024   # 160 MB: warn on mid-range phones
BLOCK_PHONE_BYTES = 512 * 1024 * 1024  # 512 MB: refuse cast to prevent Android LMK crash


def _unwrap_vessel_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Accept both Android nested ``{"config": {...}}`` and flat ``{"nLayer": ...}`` calls."""
    outer = dict(params or {})
    nested = outer.get("config")
    if isinstance(nested, dict):
        merged = dict(nested)
        for k, v in outer.items():
            if k != "config" and v is not None:
                merged[k] = v
        return merged
    return outer


def estimate_vessel(params: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate architecture and estimate RAM requirements before casting."""
    cfg_in = _unwrap_vessel_params(params)
    preset = str(cfg_in.get("preset") or "").lower()
    base = PRESETS.get(preset, PRESETS["phone"])
    n_layer = int(cfg_in.get("nLayer") or cfg_in.get("n_layer") or base["n_layer"])
    n_head = int(cfg_in.get("nHead") or cfg_in.get("n_head") or base["n_head"])
    n_embd = int(cfg_in.get("nEmbd") or cfg_in.get("n_embd") or base["n_embd"])
    block_size = int(cfg_in.get("blockSize") or cfg_in.get("block_size") or base["block_size"])
    vocab_size = int(cfg_in.get("vocabSize") or cfg_in.get("vocab_size") or base["vocab_size"])
    ffn_mult = float(cfg_in.get("ffnMult") or cfg_in.get("ffn_mult") or 4.0)

    if n_layer < 1 or n_head < 1 or n_embd < 1 or block_size < 4:
        raise ValueError("Model dimensions must all be positive integers (context >= 4).")
    if n_embd % n_head != 0:
        raise ValueError(f"Embedding size ({n_embd}) must be divisible by head count ({n_head}).")
    vocab_size = max(262, vocab_size)

    n_params = estimate_params(n_layer, n_head, n_embd, block_size, vocab_size, ffn_mult)
    # Parameter states (16 B/param) + activation working set for batch=2
    act_bytes = 2 * block_size * n_embd * n_layer * 48
    train_bytes = estimate_train_bytes(n_params) + act_bytes

    blocked = train_bytes > BLOCK_PHONE_BYTES
    warning: str | None = None
    if blocked:
        warning = (
            f"Estimated training memory ({train_bytes // (1024 * 1024)} MB) exceeds safe mobile limits. "
            "Reduce layers, embedding size, or context length."
        )
    elif train_bytes > WARN_PHONE_BYTES:
        warning = (
            f"Large vessel (~{train_bytes // (1024 * 1024)} MB training footprint): "
            "use batch size 1–2 on memory-constrained phones."
        )

    return {
        "params": n_params,
        "trainBytes": train_bytes,
        "vocabSize": vocab_size,
        "warning": warning,
        "blocked": blocked,
    }


class Trainer:
    """Owns the active GPT model, tokenizer, training streams, optimizer & checkpoints."""

    def __init__(
        self,
        data_dir: str | Path,
        corpus: Corpus,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_model_synced: Callable[[Path, Path], None] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.ckpts_dir = self.data_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)

        self.corpus = corpus
        self._emit = emit_event or (lambda _e, _d: None)
        self._on_model_synced = on_model_synced

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._train_thread: threading.Thread | None = None

        self.cfg: ModelConfig | None = None
        self.model: GPT | None = None
        self.tokenizer: BPETokenizer | None = None
        self.optimizer: AdamW | None = None
        self.train_stream: TokenStream | None = None
        self.val_stream: TokenStream | None = None
        self._stream_corpus_revision: int = -1

        # Native C++ engine bridge for sample generation during training
        self._native = NativeInferenceBridge()

        # Persistent metrics & run state
        self.step: int = 0
        self.best_loss: float | None = None
        self.val_loss: float | None = None
        self.best_val_loss: float | None = None
        self.grad_norm: float | None = None
        self.grad_clip: float = 1.0
        self.sampler_mode: str = "random"
        self.boundaries_mode: str = "respect"
        self.balance_mode: str = "off"
        self.lesson_loss_mode: str = "both"
        self.tokens_processed: int = 0
        self.loss_tokens_processed: int = 0
        self.sequences_sampled: int = 0
        self.train_seconds: float = 0.0
        self.train_tps: float = 0.0
        self.gen_tps: float = 0.0
        self.passes: int = 0
        self.last_run: dict[str, Any] | None = None

        self.casting: bool = False
        self.cast_progress: float | None = None
        self.cast_stage: str = ""

        self.running: bool = False
        self.starting: bool = False
        self.until_step: int = 0
        self.started_at_ms: int = 0
        self.stop_requested: bool = False
        self.outcome: str | None = None

        self.losses: list[float] = []
        self.validations: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.train_log: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.cfg is not None

    @property
    def active_model_path(self) -> Path:
        return self.models_dir / "active.emb"

    @property
    def active_tokenizer_path(self) -> Path:
        return self.models_dir / "active.tok.json"

    @property
    def active_optimizer_path(self) -> Path:
        return self.models_dir / "active.opt.npz"

    def _log(self, line: str) -> None:
        entry = {"at": int(time.time() * 1000), "line": str(line)}
        self.train_log.append(entry)
        if len(self.train_log) > 400:
            self.train_log = self.train_log[-400:]
        self._emit("train.log", entry)

    def _sync_active_files(self) -> tuple[Path, Path]:
        """Atomically persist active `.emb` + `.tok.json` (+ `.opt.npz`) for C++ inference & crash recovery."""
        with self._lock:
            if not self.ready or self.cfg is None or self.model is None or self.tokenizer is None:
                raise RuntimeError("No cast vessel is active.")
            mpath = self.active_model_path
            tpath = self.active_tokenizer_path
            save_model(
                mpath,
                self.cfg,
                self.model.params,
                meta={"step": self.step, "tokenizer": self.tokenizer.to_dict()},
            )
            tmp_tok = str(tpath) + ".tmp"
            with open(tmp_tok, "w", encoding="utf-8") as fh:
                fh.write(self.tokenizer.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_tok, tpath)
            if self.optimizer is not None and self.optimizer.t > 0:
                try:
                    self.optimizer.save_npz(self.active_optimizer_path)
                except Exception:
                    pass
            if self._native.available:
                try:
                    self._native.load(mpath, tpath)
                except Exception:
                    pass
        if self._on_model_synced is not None:
            try:
                self._on_model_synced(mpath, tpath)
            except Exception:
                pass
        return mpath, tpath

    def rebuild_streams_if_needed(
        self,
        *,
        mask_lesson_prompts: bool | None = None,
        balance: str | None = None,
    ) -> None:
        """Rebuild token streams from SQLite corpus whenever corpus or stream settings change."""
        with self._lock:
            if self.tokenizer is None:
                return
            mask_lp = (self.lesson_loss_mode == "response") if mask_lesson_prompts is None else mask_lesson_prompts
            bal = self.balance_mode if balance is None else balance
            rev = self.corpus.revision
            meta_rows = self.corpus.id_kind_pairs()
            if not meta_rows:
                self.train_stream = None
                self.val_stream = None
                self._stream_corpus_revision = rev
                return
            holdout = compute_holdout_ids(meta_rows, holdout_frac=0.05)
            t_stream, v_stream = build_train_val_streams(
                self.corpus.iter_documents(batch_size=64),
                self.tokenizer,
                holdout_ids=holdout,
                mask_lesson_prompts=mask_lp,
                balance=bal,
            )
            self.train_stream = t_stream
            self.val_stream = v_stream
            self._stream_corpus_revision = rev

    # ------------------------------------------------------------------
    # Cast vessel
    # ------------------------------------------------------------------
    def cast(self, params: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        if self.running or self.starting:
            raise ValueError("Stop training before recasting the vessel.")
        est = estimate_vessel(params)
        if est["blocked"]:
            raise ValueError(est["warning"] or "Vessel configuration exceeds memory limits.")
        if self.corpus.count() == 0:
            raise ValueError("Add at least one document to Fuel before casting a vessel.")

        cfg_in = _unwrap_vessel_params(params)
        preset = str(cfg_in.get("preset") or "").lower()
        base = PRESETS.get(preset, PRESETS["phone"])
        n_layer = int(cfg_in.get("nLayer") or cfg_in.get("n_layer") or base["n_layer"])
        n_head = int(cfg_in.get("nHead") or cfg_in.get("n_head") or base["n_head"])
        n_embd = int(cfg_in.get("nEmbd") or cfg_in.get("n_embd") or base["n_embd"])
        block_size = int(cfg_in.get("blockSize") or cfg_in.get("block_size") or base["block_size"])
        target_vocab = int(cfg_in.get("vocabSize") or cfg_in.get("vocab_size") or base["vocab_size"])
        ffn_mult = float(cfg_in.get("ffnMult") or cfg_in.get("ffn_mult") or 4.0)
        dropout = float(cfg_in.get("dropout", base.get("dropout", 0.0)))
        base_lr = float(params.get("lr", cfg_in.get("lr", 3e-4)))

        # Optional deterministic initialization. Without it every cast draws fresh
        # random weights, which is fine for the app but makes a run impossible to
        # reproduce; tests and debugging pass an explicit seed.
        seed_param = params.get("seed", cfg_in.get("seed"))
        if seed_param is not None:
            try:
                seed = int(seed_param)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"seed must be an integer, got {seed_param!r}") from exc

        # Preserve prior trained model automatically before recasting
        if self.ready and self.step > 0:
            try:
                self.save_checkpoint("pre-recast")
            except Exception:
                pass

        self.casting = True
        try:
            def _stage(stage: str, ratio: float) -> None:
                self.cast_stage = stage
                self.cast_progress = ratio
                self._emit("cast.progress", {"stage": stage, "ratio": ratio})
                self._emit("model.castProgress", self.status())

            _stage("corpus", 0.08)
            n_docs = self.corpus.count()
            _stage("tokenizer", 0.25)
            tok = train_bpe(
                self.corpus.iter_texts(batch_size=64),
                vocab_size=target_vocab,
                min_pair_count=2,
                total_hint=n_docs,
            )

            _stage("stream", 0.60)
            rev = self.corpus.revision
            meta_rows = self.corpus.id_kind_pairs()
            holdout = compute_holdout_ids(meta_rows, holdout_frac=0.05)
            t_stream, v_stream = build_train_val_streams(
                self.corpus.iter_documents(batch_size=64),
                tok,
                holdout_ids=holdout,
                mask_lesson_prompts=(self.lesson_loss_mode == "response"),
                balance=self.balance_mode,
            )

            _stage("model", 0.88)
            cfg = ModelConfig(
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                block_size=block_size,
                vocab_size=tok.vocab_size,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
            gpt = GPT(cfg, seed=seed)
            opt = AdamW(gpt.params, lr=base_lr)

            with self._lock:
                self.cfg = cfg
                self.tokenizer = tok
                self.model = gpt
                self.optimizer = opt
                self.train_stream = t_stream
                self.val_stream = v_stream
                self._stream_corpus_revision = rev
                self.step = 0
                self.best_loss = None
                self.val_loss = None
                self.best_val_loss = None
                self.grad_norm = None
                self.tokens_processed = 0
                self.loss_tokens_processed = 0
                self.sequences_sampled = 0
                self.train_seconds = 0.0
                self.passes = 0
                self.losses.clear()
                self.validations.clear()
                self.samples.clear()
                if self.active_optimizer_path.exists():
                    try:
                        self.active_optimizer_path.unlink()
                    except OSError:
                        pass
                self._sync_active_files()

            self._log(
                f"Cast v3 vessel: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}D ctx={cfg.block_size} "
                f"vocab={cfg.vocab_size} ({cfg.n_params:,} params, {t_stream.n_tokens:,} train tokens, "
                f"{v_stream.n_tokens:,} val tokens)."
            )
        finally:
            self.casting = False
            self.cast_progress = None
            self.cast_stage = ""

        st = self.status()
        self._emit("model.cast", st)
        return {**st, "model": st, "corpus": self.corpus.summary()}

    # ------------------------------------------------------------------
    # Validation (strictly non-mutating)
    # ------------------------------------------------------------------
    def evaluate_validation(self, max_windows: int = 16) -> dict[str, Any] | None:
        """Evaluate cross-entropy loss on held-out validation windows without modifying weights."""
        with self._lock:
            if not self.ready or self.model is None or self.cfg is None:
                return None
            if self.val_stream is None or self.val_stream.n_tokens < 2:
                return None
            stream = self.val_stream
            T = self.cfg.block_size
            B = 2
            batcher = Batcher(stream, T, B, seed=1337, sampler="epoch", boundaries=self.boundaries_mode)
            n_avail = max(1, math.ceil(stream.n_tokens / max(1, T * B)))
            n_eval = min(max_windows, n_avail)
            losses: list[float] = []
            for _ in range(n_eval):
                x, y, doc_ids = batcher.batch()
                if not np.any(y != IGNORE_INDEX):
                    continue
                _, loss = self.model.forward(x, doc_ids=doc_ids, targets=y, dropout=0.0)
                self.model._cache = None
                if loss is not None and math.isfinite(loss):
                    losses.append(float(loss))
            if not losses:
                return None
            mean_loss = float(np.mean(losses))
            if self.best_val_loss is None or mean_loss < self.best_val_loss:
                self.best_val_loss = mean_loss
            self.val_loss = mean_loss
            ppl = float(math.exp(min(20.0, mean_loss)))
            point = {
                "step": self.step,
                "loss": mean_loss,
                "bestLoss": self.best_val_loss,
                "ppl": ppl,
                "windows": len(losses),
            }
            self.validations.append(point)
            if len(self.validations) > 200:
                self.validations = self.validations[-200:]
            return point

    # ------------------------------------------------------------------
    # Training loop (synchronous step runner + background thread launcher)
    # ------------------------------------------------------------------
    def _write_autosave(self) -> dict[str, Any] | None:
        """Persist the autosave checkpoint and tell the UI about it.

        Returning ``None`` (instead of raising) keeps a failed autosave from
        killing an otherwise healthy training run, while the emit lets the
        Compose "Autosave written" notice fire after every periodic save and
        at the end of each run.
        """
        try:
            info = self.save_checkpoint("autosave")
        except Exception:
            return None
        self._emit("train.autosave", info)
        return info

    def train_sync(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run ``steps`` synchronously (used inside the worker thread and direct tests)."""
        opts = options or {}
        if not self.ready or self.model is None or self.cfg is None or self.tokenizer is None:
            raise ValueError("Cast a vessel before starting training.")

        steps = max(1, int(opts.get("steps", 150)))
        batch_size = max(1, min(32, int(opts.get("batch", 2))))
        base_lr = float(opts.get("lr", 3e-4))
        sample_every = max(0, int(opts.get("every", 25)))
        val_every = max(0, int(opts.get("validateEvery", 50)))
        grad_clip = float(opts.get("gradClip", 1.0))
        sampler_mode = str(opts.get("sampler") or "random")
        boundaries = str(opts.get("boundaries") or "respect")
        balance = str(opts.get("balance") or "off")
        lesson_loss = str(opts.get("lessonLoss") or "both")
        sample_prompt = str(opts.get("samplePrompt") or "")

        need_rebuild = (
            self.train_stream is None
            or self.train_stream.n_tokens < 2
            or self.corpus.revision != self._stream_corpus_revision
            or balance != self.balance_mode
            or lesson_loss != self.lesson_loss_mode
        )
        self.grad_clip = grad_clip
        self.sampler_mode = sampler_mode
        self.boundaries_mode = boundaries
        self.balance_mode = balance
        self.lesson_loss_mode = lesson_loss
        if need_rebuild:
            self.rebuild_streams_if_needed(
                mask_lesson_prompts=(lesson_loss == "response"),
                balance=balance,
            )

        if self.train_stream is None or self.train_stream.n_tokens < 2:
            raise ValueError("The corpus produced no training tokens. Add more text to Fuel.")

        if self.optimizer is None:
            self.optimizer = AdamW(self.model.params, lr=base_lr)
        else:
            self.optimizer.set_lr(base_lr)

        start_step = self.step
        target_step = start_step + steps
        started_iso = datetime.now(timezone.utc).isoformat()
        self._stop_event.clear()
        self.stop_requested = False
        self.starting = False
        self.running = True
        self.until_step = target_step
        self.started_at_ms = int(time.time() * 1000)
        self.outcome = None

        schedule = LRSchedule.for_run(base_lr, steps)
        batcher = Batcher(
            self.train_stream,
            block_size=self.cfg.block_size,
            batch_size=batch_size,
            seed=(start_step * 7919 + 42) & 0x7FFFFFFF,
            sampler=sampler_mode,
            boundaries=boundaries,
        )
        drop_rng = np.random.default_rng(start_step + 17)
        ema: float | None = None
        completed = 0
        outcome = "completed"
        error_msg: str | None = None

        self._log(
            f"Fire started at step {start_step} -> {target_step} "
            f"(batch={batch_size}, lr={base_lr:g}, clip={grad_clip:g}, boundaries={boundaries})"
        )

        try:
            for local_i in range(steps):
                if self._stop_event.is_set():
                    outcome = "doused"
                    break

                step_t0 = time.perf_counter()
                lr = schedule.lr_at(local_i)

                with self._lock:
                    self.optimizer.set_lr(lr)
                    x, y, doc_ids = batcher.batch()
                    self.model.zero_grad()
                    _, loss = self.model.forward(
                        x,
                        doc_ids=doc_ids,
                        targets=y,
                        dropout=self.cfg.dropout,
                        rng=drop_rng,
                    )
                    if loss is None or not math.isfinite(loss):
                        self.model._cache = None
                        self._log(f"! step {self.step + 1}: non-finite loss skipped")
                        continue

                    grads = self.model.backward()
                    self.model._cache = None
                    if grad_has_nan(grads.values()):
                        self._log(f"! step {self.step + 1}: NaN/Inf gradient refused")
                        continue

                    gnorm, clipped = clip_grad_norm(grads, grad_clip)
                    self.optimizer.step(grads)

                    self.step += 1
                    completed += 1
                    step_sec = max(1e-6, time.perf_counter() - step_t0)
                    step_tokens = int(batch_size * self.cfg.block_size)
                    step_loss_tokens = int(np.count_nonzero(y != IGNORE_INDEX))

                    self.tokens_processed += step_tokens
                    self.loss_tokens_processed += step_loss_tokens
                    self.sequences_sampled += batch_size
                    self.train_seconds += step_sec
                    self.train_tps = step_tokens / step_sec
                    self.grad_norm = gnorm
                    self.passes = int(self.tokens_processed // max(1, self.train_stream.n_tokens))

                    f_loss = float(loss)
                    ema = f_loss if ema is None else (0.9 * ema + 0.1 * f_loss)
                    if self.best_loss is None or f_loss < self.best_loss:
                        self.best_loss = f_loss
                    self.losses.append(f_loss)
                    if len(self.losses) > 600:
                        self.losses = self.losses[-600:]

                progress_payload = {
                    "step": self.step,
                    "until": target_step,
                    "startStep": start_step,
                    "loss": f_loss,
                    "ema": ema,
                    "bestLoss": self.best_loss,
                    "lr": lr,
                    "gradNorm": gnorm,
                    "clipped": clipped,
                    "ms": step_sec * 1000.0,
                    "trainTps": self.train_tps,
                    "tokensProcessed": self.tokens_processed,
                    "lossTokensProcessed": self.loss_tokens_processed,
                    "sequencesSampled": self.sequences_sampled,
                    "trainSeconds": self.train_seconds,
                    "valLoss": self.val_loss,
                    "bestValLoss": self.best_val_loss,
                    "passes": self.passes,
                }
                self._emit("train.progress", progress_payload)

                if val_every > 0 and (completed % val_every == 0 or local_i == steps - 1):
                    vpt = self.evaluate_validation()
                    if vpt is not None:
                        self._log(
                            f"  step {self.step}: train={f_loss:.4f} val={vpt['loss']:.4f} ppl={vpt['ppl']:.2f}"
                        )
                        self._emit("train.validation", vpt)

                if sample_every > 0 and (completed % sample_every == 0):
                    sample = self.generate_sample(prompt=sample_prompt, max_new_tokens=48)
                    entry = {
                        "step": self.step,
                        "text": sample.text or "(silence)",
                        "tokensPerSec": sample.tokens_per_sec,
                    }
                    self.samples.insert(0, entry)
                    self.samples = self.samples[:12]
                    self._emit("train.sample", entry)

                if completed % 50 == 0:
                    self._write_autosave()

        except Exception as exc:
            outcome = "failed"
            error_msg = str(exc)
            self._log(f"! Training error: {exc}")
            raise
        finally:
            self.running = False
            self.starting = False
            self.stop_requested = False
            self.outcome = outcome
            self.last_run = {
                "startedAt": started_iso,
                "endedAt": datetime.now(timezone.utc).isoformat(),
                "startStep": start_step,
                "requestedSteps": steps,
                "completedSteps": completed,
                "batchSize": batch_size,
                "baseLr": base_lr,
                "gradientClip": grad_clip,
                "sampler": sampler_mode,
                "boundaries": boundaries,
                "balanceMode": balance,
                "lessonLoss": "response" if lesson_loss == "response" else "prompt+response",
                "validationEvery": val_every,
                "outcome": outcome,
            }
            if self.ready and completed > 0:
                try:
                    self._sync_active_files()
                except Exception:
                    pass
                self._write_autosave()
            self._log(f"Fire {outcome} at step {self.step} (best loss {self.best_loss or 0.0:.4f}).")
            done_payload = {
                "outcome": outcome,
                "error": error_msg,
                "step": self.step,
                "completedSteps": completed,
                "bestLoss": self.best_loss,
                "bestValLoss": self.best_val_loss,
                "model": self.status(),
                "corpus": self.corpus.summary(),
            }
            self._emit("train.finished", done_payload)
            self._emit("train.done", done_payload)

        return self.last_run

    def start_background(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.running or self.starting:
                raise ValueError("Training is already running.")
            if not self.ready:
                raise ValueError("Cast a vessel before starting training.")
            if self.corpus.count() == 0:
                raise ValueError("Add text or a starter pack to Fuel before firing.")
            opts = dict(options or {})
            steps = max(1, int(opts.get("steps", 150)))
            self.starting = True
            self.until_step = self.step + steps
            self._stop_event.clear()

            def _run() -> None:
                try:
                    self.train_sync(opts)
                except Exception as exc:
                    self.running = False
                    self.starting = False
                    self._emit(
                        "train.finished",
                        {
                            "outcome": "failed",
                            "error": str(exc),
                            "step": self.step,
                            "completedSteps": 0,
                            "model": self.status(),
                            "corpus": self.corpus.summary(),
                        },
                    )
                    self._emit("train.error", {"message": str(exc)})

            t = threading.Thread(target=_run, name="ember-trainer", daemon=True)
            self._train_thread = t
            t.start()
        return {"started": True, "step": self.step, "until": self.until_step}

    def stop(self) -> dict[str, Any]:
        self.stop_requested = True
        self._stop_event.set()
        return {"stopping": True, "step": self.step}

    # ------------------------------------------------------------------
    # Sample & inference generation interface
    # ------------------------------------------------------------------
    def generate_sample(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        seed: int | None = None,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Generate text via the native C++ inference bridge (or Python reference fallback)."""
        with self._lock:
            if not self.ready or self.model is None or self.tokenizer is None:
                raise ValueError("Cast or load a vessel before generating text.")
            self._sync_active_files()

        if self._native.available:
            res = self._native.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_on_eos=stop_on_eos,
            )
            self.gen_tps = res.tokens_per_sec
            return res

        with self._lock:
            res = reference_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_on_eos=stop_on_eos,
            )
            self.gen_tps = res.tokens_per_sec
            return res

    # ------------------------------------------------------------------
    # Checkpoints (atomic .emb + .tok.json + .state.json + .opt.npz)
    # ------------------------------------------------------------------
    def _ckpt_paths(self, name: str) -> tuple[Path, Path, Path, Path]:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "checkpoint"))[:80] or "checkpoint"
        return (
            self.ckpts_dir / f"{safe}.emb",
            self.ckpts_dir / f"{safe}.tok.json",
            self.ckpts_dir / f"{safe}.state.json",
            self.ckpts_dir / f"{safe}.opt.npz",
        )

    def save_checkpoint(self, name: str = "checkpoint") -> dict[str, Any]:
        with self._lock:
            if not self.ready or self.cfg is None or self.model is None or self.tokenizer is None:
                raise ValueError("No active vessel to checkpoint.")
            emb_p, tok_p, st_p, opt_p = self._ckpt_paths(name)
            save_model(
                emb_p,
                self.cfg,
                self.model.params,
                meta={"step": self.step, "name": name, "tokenizer": self.tokenizer.to_dict()},
            )
            tmp_tok = str(tok_p) + ".tmp"
            with open(tmp_tok, "w", encoding="utf-8") as fh:
                fh.write(self.tokenizer.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_tok, tok_p)

            if self.optimizer is not None and self.optimizer.t > 0:
                try:
                    self.optimizer.save_npz(opt_p)
                except Exception:
                    pass

            state_obj = {
                "name": name,
                "step": self.step,
                "best_loss": self.best_loss,
                "val_loss": self.val_loss,
                "best_val_loss": self.best_val_loss,
                "tokens_processed": self.tokens_processed,
                "loss_tokens_processed": self.loss_tokens_processed,
                "sequences_sampled": self.sequences_sampled,
                "train_seconds": self.train_seconds,
                "passes": self.passes,
                "losses": self.losses[-400:],
                "validations": self.validations[-120:],
                "last_run": self.last_run,
                "saved_at": int(time.time() * 1000),
            }
            tmp_st = str(st_p) + ".tmp"
            with open(tmp_st, "w", encoding="utf-8") as fh:
                json.dump(state_obj, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_st, st_p)

            total_bytes = emb_p.stat().st_size + tok_p.stat().st_size + st_p.stat().st_size
            return {
                "name": emb_p.stem,
                "step": self.step,
                "bytes": total_bytes,
                "mtime": int(emb_p.stat().st_mtime * 1000),
            }

    def export_checkpoint_dict(self, name: str, include_corpus: bool = False) -> dict[str, Any]:
        """Read a named checkpoint into a portable v4 JSON dict WITHOUT mutating live model state."""
        emb_p, tok_p, st_p, _ = self._ckpt_paths(name)
        if not emb_p.exists() or not tok_p.exists():
            raise ValueError(f"Checkpoint '{name}' was not found.")
        cfg, params, meta = load_model(emb_p)
        tok = BPETokenizer.load(tok_p)
        st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.exists() else {}
        step_val = int(st.get("step", meta.get("step", 0)))
        brain: dict[str, Any] = {
            "kind": BRAIN_KIND,
            "format": "ember-brain",
            "version": BRAIN_VERSION,
            "spec": "v3",
            "name": emb_p.stem,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "step": step_val,
            "config": cfg.to_dict(),
            "tokenizer": tok.to_dict(),
            "weights": model_to_dict(cfg, params),
            "training_state": {
                "step": step_val,
                "best_loss": st.get("best_loss"),
                "val_loss": st.get("val_loss"),
                "best_val_loss": st.get("best_val_loss"),
                "tokens_processed": int(st.get("tokens_processed", 0)),
                "loss_tokens_processed": int(st.get("loss_tokens_processed", 0)),
                "sequences_sampled": int(st.get("sequences_sampled", 0)),
                "train_seconds": float(st.get("train_seconds", 0.0)),
                "losses": list(st.get("losses", []))[-400:],
                "validations": list(st.get("validations", []))[-120:],
            },
        }
        if include_corpus:
            brain["corpusDocs"] = self.corpus.export_docs()
        return brain

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        if self.running or self.starting:
            raise ValueError("Stop training before loading a checkpoint.")
        emb_p, tok_p, st_p, opt_p = self._ckpt_paths(name)
        if not emb_p.exists() or not tok_p.exists():
            raise ValueError(f"Checkpoint '{name}' was not found.")
        cfg, params, meta = load_model(emb_p)
        tok = BPETokenizer.load(tok_p)
        st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.exists() else {}
        with self._lock:
            self.cfg = cfg
            self.tokenizer = tok
            self.model = GPT(cfg, params=params)
            self.optimizer = AdamW(self.model.params, lr=3e-4)
            if opt_p.exists():
                try:
                    self.optimizer.load_npz(opt_p)
                except Exception:
                    pass
            self.step = int(st.get("step", meta.get("step", 0)))
            self.best_loss = st.get("best_loss")
            self.val_loss = st.get("val_loss")
            self.best_val_loss = st.get("best_val_loss")
            self.tokens_processed = int(st.get("tokens_processed", 0))
            self.loss_tokens_processed = int(st.get("loss_tokens_processed", 0))
            self.sequences_sampled = int(st.get("sequences_sampled", 0))
            self.train_seconds = float(st.get("train_seconds", 0.0))
            self.passes = int(st.get("passes", 0))
            self.losses = list(st.get("losses", []))
            self.validations = list(st.get("validations", []))
            self.last_run = st.get("last_run")
            self.rebuild_streams_if_needed()
            self._sync_active_files()
        self._log(f"Loaded checkpoint '{name}' (step {self.step}).")
        cur = self.status()
        return {**cur, "model": cur, "corpus": self.corpus.summary()}

    def delete_checkpoint(self, name: str) -> bool:
        emb_p, tok_p, st_p, opt_p = self._ckpt_paths(name)
        removed = False
        for p in (emb_p, tok_p, st_p, opt_p):
            if p.exists():
                p.unlink()
                removed = True
        return removed

    def list_checkpoints(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for emb_p in sorted(self.ckpts_dir.glob("*.emb"), key=lambda p: p.stat().st_mtime, reverse=True):
            name = emb_p.stem
            tok_p = self.ckpts_dir / f"{name}.tok.json"
            st_p = self.ckpts_dir / f"{name}.state.json"
            step = 0
            if st_p.exists():
                try:
                    step = int(json.loads(st_p.read_text(encoding="utf-8")).get("step", 0))
                except Exception:
                    step = 0
            size = emb_p.stat().st_size + (tok_p.stat().st_size if tok_p.exists() else 0)
            out.append(
                {
                    "name": name,
                    "bytes": size,
                    "mtime": int(emb_p.stat().st_mtime * 1000),
                    "step": step,
                }
            )
        return out

    def probe_autosave(self) -> dict[str, Any] | None:
        emb_p, tok_p, st_p, _ = self._ckpt_paths("autosave")
        if not emb_p.exists() or not tok_p.exists():
            return None
        try:
            cfg, _, meta = load_model(emb_p)
            st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.exists() else {}
            return {
                "name": "autosave",
                "step": int(st.get("step", meta.get("step", 0))),
                "tokenizer": TOKENIZER_TYPE,
                "mtime": int(emb_p.stat().st_mtime * 1000),
                "bytes": emb_p.stat().st_size,
                "nParams": cfg.n_params,
            }
        except Exception as exc:
            return {"name": "autosave", "error": str(exc)}

    # ------------------------------------------------------------------
    # Portable Brain JSON (v4), .emb binary import, + legacy JS brain handling
    # ------------------------------------------------------------------
    def serialize_brain(self, name: str = "ember-brain", include_corpus: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self.ready or self.cfg is None or self.model is None or self.tokenizer is None:
                raise ValueError("Nothing to export — cast a vessel first.")
            brain: dict[str, Any] = {
                "kind": BRAIN_KIND,
                "format": "ember-brain",
                "version": BRAIN_VERSION,
                "spec": "v3",
                "name": name,
                "exportedAt": datetime.now(timezone.utc).isoformat(),
                "step": self.step,
                "config": self.cfg.to_dict(),
                "tokenizer": self.tokenizer.to_dict(),
                "weights": model_to_dict(self.cfg, self.model.params),
                "training_state": {
                    "step": self.step,
                    "best_loss": self.best_loss,
                    "val_loss": self.val_loss,
                    "best_val_loss": self.best_val_loss,
                    "tokens_processed": self.tokens_processed,
                    "loss_tokens_processed": self.loss_tokens_processed,
                    "sequences_sampled": self.sequences_sampled,
                    "train_seconds": self.train_seconds,
                    "losses": self.losses[-400:],
                    "validations": self.validations[-120:],
                },
            }
        if include_corpus:
            brain["corpusDocs"] = self.corpus.export_docs()
        return brain

    def import_brain_emb(
        self,
        raw: bytes,
        *,
        name: str = "imported",
        companion_tok_path: Path | None = None,
        save_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """Import a native v3 binary `.emb` file directly into the live vessel."""
        if self.running or self.starting:
            raise ValueError("Stop training before importing a model.")
        cfg, params, meta = load_model_bytes(raw)
        tok: BPETokenizer | None = None
        if isinstance(meta.get("tokenizer"), dict):
            tok = BPETokenizer.from_dict(meta["tokenizer"])
        elif companion_tok_path is not None and companion_tok_path.exists():
            tok = BPETokenizer.load(companion_tok_path)
        elif self.tokenizer is not None and self.tokenizer.vocab_size == cfg.vocab_size:
            tok = self.tokenizer
        else:
            raise ValueError(
                f"This standalone .emb file (vocab={cfg.vocab_size}) has no embedded tokenizer metadata "
                "and no matching active tokenizer. Export as a full brain or place its .tok.json alongside it."
            )
        if tok.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({tok.vocab_size}) does not match model vocab_size ({cfg.vocab_size})."
            )

        if self.ready and self.step > 0:
            try:
                self.save_checkpoint("pre-import")
            except Exception:
                pass

        with self._lock:
            self.cfg = cfg
            self.tokenizer = tok
            self.model = GPT(cfg, params=params)
            self.optimizer = AdamW(self.model.params, lr=3e-4)
            self.step = int(meta.get("step", 0))
            self.best_loss = None
            self.val_loss = None
            self.best_val_loss = None
            self.tokens_processed = 0
            self.loss_tokens_processed = 0
            self.sequences_sampled = 0
            self.train_seconds = 0.0
            self.losses = []
            self.validations = []
            self.rebuild_streams_if_needed()
            self._sync_active_files()
            ckpt_info = self.save_checkpoint(name) if save_checkpoint else None

        st = self.status()
        self._emit("model.imported", st)
        return {
            "imported": True,
            "version": BRAIN_VERSION,
            "name": str(meta.get("name") or name),
            "step": self.step,
            "nParams": cfg.n_params,
            "vocabSize": cfg.vocab_size,
            "tokenizer": TOKENIZER_TYPE,
            "corpusAdded": 0,
            "importedCorpusDocuments": 0,
            "checkpoint": ckpt_info,
            "model": st,
            "corpus": self.corpus.summary(),
        }

    def import_brain(
        self,
        data: dict[str, Any],
        *,
        name: str = "imported",
        save_checkpoint: bool = True,
    ) -> dict[str, Any]:
        if self.running or self.starting:
            raise ValueError("Stop training before importing a brain.")
        if not isinstance(data, dict):
            raise ValueError("Brain file must contain a JSON object.")

        version = int(data.get("version") or 0)
        fmt_str = str(data.get("format") or "")
        if data.get("kind") != BRAIN_KIND or version != BRAIN_VERSION:
            docs = data.get("corpusDocs") or data.get("corpus") or []
            recovered = 0
            if isinstance(docs, list) and docs:
                res = self.corpus.add_many([d for d in docs if isinstance(d, dict)])
                recovered = res["accepted"]
            suffix = (
                f" Imported {recovered} embedded training document(s) into Fuel so you can cast a fresh v3 vessel immediately."
                if recovered
                else " Your existing Fuel corpus was preserved untouched — cast a new v3 vessel to train."
            )
            raise ValueError(
                f"Incompatible legacy brain format ({fmt_str or 'legacy'} v{version or 1}): "
                "weights trained with the deprecated TensorFlow.js engine use a different schema and tokenizer "
                f"from Ember v3 (Python + native C++).{suffix}"
            )

        cfg = ModelConfig.from_dict(data["config"])
        tok = BPETokenizer.from_dict(data["tokenizer"])
        params = model_from_dict(cfg, data["weights"])
        check_params(cfg, params)

        if self.ready and self.step > 0:
            try:
                self.save_checkpoint("pre-import")
            except Exception:
                pass

        docs = data.get("corpusDocs")
        corpus_added = 0
        if isinstance(docs, list) and docs:
            res = self.corpus.add_many([d for d in docs if isinstance(d, dict)])
            corpus_added = res["accepted"]

        tstate = data.get("training_state") or {}
        with self._lock:
            self.cfg = cfg
            self.tokenizer = tok
            self.model = GPT(cfg, params=params)
            self.optimizer = AdamW(self.model.params, lr=3e-4)
            self.step = int(tstate.get("step", data.get("step", 0)))
            self.best_loss = tstate.get("best_loss")
            self.val_loss = tstate.get("val_loss")
            self.best_val_loss = tstate.get("best_val_loss")
            self.tokens_processed = int(tstate.get("tokens_processed", 0))
            self.loss_tokens_processed = int(tstate.get("loss_tokens_processed", 0))
            self.sequences_sampled = int(tstate.get("sequences_sampled", 0))
            self.train_seconds = float(tstate.get("train_seconds", 0.0))
            self.losses = list(tstate.get("losses", []))
            self.validations = list(tstate.get("validations", []))
            self.rebuild_streams_if_needed()
            self._sync_active_files()
            ckpt_info = self.save_checkpoint(name) if save_checkpoint else None

        st = self.status()
        self._emit("model.imported", st)
        return {
            "imported": True,
            "version": BRAIN_VERSION,
            "name": str(data.get("name") or name),
            "step": self.step,
            "nParams": cfg.n_params,
            "vocabSize": cfg.vocab_size,
            "tokenizer": TOKENIZER_TYPE,
            "corpusAdded": corpus_added,
            "importedCorpusDocuments": corpus_added,
            "checkpoint": ckpt_info,
            "model": st,
            "corpus": self.corpus.summary(),
        }

    # ------------------------------------------------------------------
    # Status snapshot matching Android Models.kt ModelStatus
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        ts = self.train_stream
        vs = self.val_stream
        val_ppl = float(math.exp(min(20.0, self.val_loss))) if self.val_loss is not None else None
        return {
            "ready": self.ready,
            "step": self.step,
            "bestLoss": self.best_loss,
            "bestValLoss": self.best_val_loss,
            "nParams": self.cfg.n_params if self.cfg else 0,
            "vocabSize": self.tokenizer.vocab_size if self.tokenizer else 0,
            "tokenizerType": TOKENIZER_TYPE,
            "corpusTokens": ts.n_tokens if ts else 0,
            "valTokens": vs.n_tokens if vs else 0,
            "documents": (ts.corpus_documents if ts else self.corpus.count()),
            "trainDocuments": (ts.sampled_documents if ts else 0),
            "validationDocuments": (vs.sampled_documents if vs else 0),
            "tokensProcessed": self.tokens_processed,
            "lossTokensProcessed": self.loss_tokens_processed,
            "sequencesSampled": self.sequences_sampled,
            "trainSeconds": self.train_seconds,
            "valLoss": self.val_loss,
            "valPpl": val_ppl,
            "gradNorm": self.grad_norm,
            "trainTps": self.train_tps,
            "genTps": self.gen_tps,
            "gradClip": self.grad_clip,
            "sampler": self.sampler_mode,
            "passes": self.passes,
            "streamFormat": 3,
            "docBoundaryMask": self.boundaries_mode == "respect",
            "maskBoundaryLoss": True,
            "maskLessonPrompts": self.lesson_loss_mode == "response",
            "balanceMode": self.balance_mode,
            "sourceTokens": ts.source_tokens if ts else {},
            "sourceBoosts": {k: int(round(v * 100)) for k, v in (ts.source_boosts.items() if ts else ())},
            "config": self.cfg.to_dict() if self.cfg else None,
            "lastRun": self.last_run,
            "backend": "python-numpy + native-cpp",
            "castProgress": self.cast_progress,
            "castStage": self.cast_stage,
            "castSampled": bool(ts.truncated) if ts else False,
            "castSampleDocuments": ts.sampled_documents if ts else 0,
            "castCorpusDocuments": ts.corpus_documents if ts else 0,
            "castSampleChars": ts.sampled_chars if ts else 0,
            "castCorpusChars": ts.corpus_chars if ts else 0,
        }
