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


def estimate_vessel(params: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate architecture and estimate RAM requirements before casting."""
    preset = str(params.get("preset") or "").lower()
    base = PRESETS.get(preset, PRESETS["phone"])
    n_layer = int(params.get("nLayer") or params.get("n_layer") or base["n_layer"])
    n_head = int(params.get("nHead") or params.get("n_head") or base["n_head"])
    n_embd = int(params.get("nEmbd") or params.get("n_embd") or base["n_embd"])
    block_size = int(params.get("blockSize") or params.get("block_size") or base["block_size"])
    vocab_size = int(params.get("vocabSize") or params.get("vocab_size") or base["vocab_size"])
    ffn_mult = float(params.get("ffnMult") or params.get("ffn_mult") or 4.0)

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

    def _log(self, line: str) -> None:
        entry = {"at": int(time.time() * 1000), "line": str(line)}
        self.train_log.append(entry)
        if len(self.train_log) > 400:
            self.train_log = self.train_log[-400:]
        self._emit("train.log", entry)

    def _sync_active_files(self) -> tuple[Path, Path]:
        """Atomically persist active `.emb` + `.tok.json` for native C++ inference."""
        with self._lock:
            if not self.ready or self.cfg is None or self.model is None or self.tokenizer is None:
                raise RuntimeError("No cast vessel is active.")
            mpath = self.active_model_path
            tpath = self.active_tokenizer_path
            save_model(mpath, self.cfg, self.model.params, meta={"step": self.step})
            tmp_tok = str(tpath) + ".tmp"
            with open(tmp_tok, "w", encoding="utf-8") as fh:
                fh.write(self.tokenizer.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_tok, tpath)
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
        """Rebuild token streams from SQLite corpus (e.g., after new fuel or checkpoint load)."""
        with self._lock:
            if self.tokenizer is None:
                return
            mask_lp = (self.lesson_loss_mode == "response") if mask_lesson_prompts is None else mask_lesson_prompts
            bal = self.balance_mode if balance is None else balance
            meta_rows = self.corpus.id_kind_pairs()
            if not meta_rows:
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

        preset = str(params.get("preset") or "").lower()
        base = PRESETS.get(preset, PRESETS["phone"])
        n_layer = int(params.get("nLayer") or params.get("n_layer") or base["n_layer"])
        n_head = int(params.get("nHead") or params.get("n_head") or base["n_head"])
        n_embd = int(params.get("nEmbd") or params.get("n_embd") or base["n_embd"])
        block_size = int(params.get("blockSize") or params.get("block_size") or base["block_size"])
        target_vocab = int(params.get("vocabSize") or params.get("vocab_size") or base["vocab_size"])
        ffn_mult = float(params.get("ffnMult") or params.get("ffn_mult") or 4.0)
        dropout = float(params.get("dropout", base.get("dropout", 0.0)))
        base_lr = float(params.get("lr", 3e-4))

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

            _stage("tokenizer", 0.15)
            n_docs = self.corpus.count()
            tok = train_bpe(
                self.corpus.iter_texts(batch_size=64),
                vocab_size=target_vocab,
                min_pair_count=2,
                total_hint=n_docs,
            )

            _stage("encoding", 0.55)
            meta_rows = self.corpus.id_kind_pairs()
            holdout = compute_holdout_ids(meta_rows, holdout_frac=0.05)
            t_stream, v_stream = build_train_val_streams(
                self.corpus.iter_documents(batch_size=64),
                tok,
                holdout_ids=holdout,
                mask_lesson_prompts=(self.lesson_loss_mode == "response"),
                balance=self.balance_mode,
            )

            _stage("initializing", 0.85)
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
                self._sync_active_files()

            _stage("ready", 1.0)
            self._log(
                f"Cast v3 vessel: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}D ctx={cfg.block_size} "
                f"vocab={cfg.vocab_size} ({cfg.n_params:,} params, {t_stream.n_tokens:,} train tokens, "
                f"{v_stream.n_tokens:,} val tokens)."
            )
            return self.status()
        finally:
            self.casting = False
            self.cast_progress = None
            self.cast_stage = ""

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

        self._log(
            f"Fire started at step {start_step} -> {target_step} "
            f"(batch={batch_size}, lr={base_lr:g}, clip={grad_clip:g}, boundaries={boundaries})"
        )

        try:
            for local_i in range(steps):
                if self._stop_event.is_set():
                    outcome = "stopped"
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

                if completed % 100 == 0:
                    try:
                        self.save_checkpoint("autosave")
                    except Exception:
                        pass

        except Exception as exc:
            outcome = "failed"
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
                    self.save_checkpoint("autosave")
                except Exception:
                    pass
            self._log(f"Fire {outcome} at step {self.step} (best loss {self.best_loss or 0.0:.4f}).")
            self._emit(
                "train.done",
                {
                    "outcome": outcome,
                    "step": self.step,
                    "completedSteps": completed,
                    "bestLoss": self.best_loss,
                    "bestValLoss": self.best_val_loss,
                },
            )

        return self.last_run

    def start_background(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.running or self.starting:
                raise ValueError("Training is already running.")
            if not self.ready:
                raise ValueError("Cast a vessel before starting training.")
            self.starting = True
            self._stop_event.clear()
            opts = dict(options or {})

            def _run() -> None:
                try:
                    self.train_sync(opts)
                except Exception as exc:
                    self._emit("train.error", {"message": str(exc)})

            t = threading.Thread(target=_run, name="ember-trainer", daemon=True)
            self._train_thread = t
            t.start()
        return {"started": True, "step": self.step}

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
            mpath, tpath = self._sync_active_files()

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
    # Checkpoints (atomic .emb + .tok.json + .state.json)
    # ------------------------------------------------------------------
    def _ckpt_paths(self, name: str) -> tuple[Path, Path, Path]:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "checkpoint"))[:80] or "checkpoint"
        return (
            self.ckpts_dir / f"{safe}.emb",
            self.ckpts_dir / f"{safe}.tok.json",
            self.ckpts_dir / f"{safe}.state.json",
        )

    def save_checkpoint(self, name: str = "checkpoint") -> dict[str, Any]:
        with self._lock:
            if not self.ready or self.cfg is None or self.model is None or self.tokenizer is None:
                raise ValueError("No active vessel to checkpoint.")
            emb_p, tok_p, st_p = self._ckpt_paths(name)
            save_model(emb_p, self.cfg, self.model.params, meta={"step": self.step, "name": name})
            tmp_tok = str(tok_p) + ".tmp"
            with open(tmp_tok, "w", encoding="utf-8") as fh:
                fh.write(self.tokenizer.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_tok, tok_p)

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

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        if self.running or self.starting:
            raise ValueError("Stop training before loading a checkpoint.")
        emb_p, tok_p, st_p = self._ckpt_paths(name)
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
        return self.status()

    def delete_checkpoint(self, name: str) -> bool:
        emb_p, tok_p, st_p = self._ckpt_paths(name)
        removed = False
        for p in (emb_p, tok_p, st_p):
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
        emb_p, tok_p, st_p = self._ckpt_paths("autosave")
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
    # Portable Brain JSON (v4) + explicit legacy JS brain handling
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
            # Recover reusable corpus docs if the legacy JS brain bundled them,
            # while explicitly telling the user why legacy weights cannot load!
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

        return {
            "imported": True,
            "name": str(data.get("name") or name),
            "step": self.step,
            "nParams": cfg.n_params,
            "vocabSize": cfg.vocab_size,
            "tokenizer": TOKENIZER_TYPE,
            "corpusAdded": corpus_added,
            "checkpoint": ckpt_info,
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
