"""Unified Ember v3 JSON-RPC backend (Android Chaquopy + Desktop HTTP + Tests)."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corpus import Corpus
from .forager import Forager
from .gguf_export import EMBER_MODEL_FORMAT, GgufExportError, validate_gguf_file
from .ingest import ImportManager
from .inference import EngineConfig, InferenceError, get_engine
from .model import FORMAT_MAGIC
from .packs import get_pack
from .teachers import WINDOWS, TeacherDesk, github_head, github_pull_brain, github_push_brain
from .trainer import Trainer, estimate_vessel

PACK_TITLES = {
    "starter": "Public-domain starter",
    "topic": "Topic-training pack",
    "translation": "Translation lessons",
    "research": "Research / QA lessons",
    "skills": "General-skills lessons",
}


def format_task_prompt(
    corpus: Corpus,
    prompt: str,
    *,
    task: str = "continue",
    language: str = "Spanish",
    subreddit: str = "",
) -> str:
    """Format user prompt according to the selected Voice task mode."""
    p = (prompt or "").strip()
    t = (task or "continue").strip().lower()
    if t == "translate":
        lang = (language or "Spanish").strip() or "Spanish"
        return f"### Task: Translate to {lang}\n### Input:\n{p}\n\n### Response:\n"
    if t == "summarize":
        return f"### Task: Summarize\n### Input:\n{p}\n\n### Response:\n"
    if t == "documents":
        passages = corpus.retrieve_passages(p, k=3, max_chars=1000)
        ctx = "\n\n---\n\n".join(passages) if passages else "(No matching passages in corpus.)"
        return f"### Context:\n{ctx}\n\n### Question:\n{p}\n\n### Response:\n"
    if subreddit.strip():
        sub = subreddit.strip().lstrip("r/")
        return f"[r/{sub}] {p}" if p else f"[r/{sub}]\n"
    return prompt or ""


class EmberBackend:
    """Complete Python backend owning corpus, ingestion, training, checkpoints, and sync."""

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: str | Path | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_model_synced: Callable[[str, str], None] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_dir / "cache")
        self.exports_dir = self.cache_dir / "exports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self._event_cb = on_event
        self._model_sync_cb = on_model_synced

        self.corpus = Corpus(self.data_dir)
        self.importer = ImportManager(self.corpus)
        self.trainer = Trainer(
            self.data_dir,
            self.corpus,
            emit_event=self.emit,
            on_model_synced=self._handle_model_synced,
        )
        self.forager = Forager(
            self.corpus,
            emit_event=self.emit,
            auto_fire_cb=self._maybe_auto_fire_after_forage,
        )
        self.teachers = TeacherDesk(self.corpus, emit_event=self.emit)
        self.autosave = self.trainer.probe_autosave()

        # Automatically restore active.emb + active.tok.json (+ active.opt.npz) across app restarts
        if self.trainer.active_model_path.exists() and self.trainer.active_tokenizer_path.exists():
            try:
                from .model import Llama, load_model
                from .optim import AdamW
                from .tokenizer import BPETokenizer

                cfg, params, meta = load_model(self.trainer.active_model_path)
                tok = BPETokenizer.from_dict(meta["tokenizer"]) if meta.get("tokenizer") else BPETokenizer.load(self.trainer.active_tokenizer_path)
                self.trainer.cfg = cfg
                self.trainer.tokenizer = tok
                self.trainer.model = Llama(cfg, params=params)
                self.trainer.optimizer = AdamW(self.trainer.model.params, lr=3e-4)
                opt_name=meta.get("optimizer_file")
                if opt_name:
                    if Path(opt_name).name != opt_name: raise ValueError("invalid optimizer reference")
                    self.trainer.optimizer.load_npz(self.trainer.models_dir/opt_name)
                elif self.trainer.active_optimizer_path.exists():
                    self.trainer.optimizer.load_npz(self.trainer.active_optimizer_path)
                state=meta.get("training_state",{})
                self.trainer.schedule_state=state.get("schedule",{})
                self.trainer.batcher_state=state.get("batcher",{})
                self.trainer.step = int(meta.get("step", 0))
                self.trainer.rebuild_streams_if_needed()
                # Restore the GGUF the native engine loads (weights come from
                # active.emb, so the export is deterministic and cheap).
                self.trainer.mark_gguf_dirty()
                try:
                    self.trainer.sync_gguf()
                except GgufExportError:
                    pass
                self._handle_model_synced(self.trainer.active_model_path, self.trainer.active_tokenizer_path)
            except Exception as exc:
                self.trainer._log(f"! Active checkpoint recovery failed: {exc}; corpus preserved.")

    def _maybe_auto_fire_after_forage(self) -> None:
        if self.trainer.ready and not self.trainer.running and not self.trainer.starting:
            self.trainer.start_background({"steps": 40, "batch": 2, "lr": 3e-4})

    def _handle_model_synced(self, model_path: Path, tok_path: Path) -> None:
        if self._model_sync_cb is not None:
            try:
                self._model_sync_cb(str(model_path), str(tok_path))
            except Exception:
                pass

    def emit(self, name: str, data: dict[str, Any] | None = None) -> None:
        if self._event_cb is not None:
            try:
                self._event_cb(name, data or {})
            except Exception:
                pass

    def close(self) -> None:
        self.trainer.stop()
        self.forager.stop()
        self.teachers.stop_auto()
        # Workers can be saving a checkpoint or accepting a document. Never close
        # SQLite beneath them during an app/runtime restart.
        for worker in (self.trainer._train_thread, self.forager._thread, getattr(self.teachers, "_thread", None)):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=60)
                if worker.is_alive():
                    raise RuntimeError("Background work is still stopping; runtime storage was kept open.")
        # Release native model memory; the process-wide engine object stays alive
        # so a later restart can reload without re-initializing llama.cpp.
        try:
            self.trainer.unload_engine_model()
        except Exception:
            pass
        self.corpus.close()

    # ------------------------------------------------------------------
    # Full status snapshot
    # ------------------------------------------------------------------
    def full_status(self) -> dict[str, Any]:
        gh = self.corpus.get_kv("github") or {}
        return {
            "ready": True,
            "backend": "python-numpy + llama.cpp",
            "backendTried": ["python-numpy", "native-cpp"],
            "runtime": f"Python {platform.python_version()} · NumPy {np.__version__}",
            "model": self.trainer.status(),
            "activeModelPath": str(self.trainer.active_model_path) if self.trainer.ready else "",
            "activeTokenizerPath": str(self.trainer.active_tokenizer_path) if self.trainer.ready else "",
            "corpus": self.corpus.summary(),
            "checkpoints": self.trainer.list_checkpoints(),
            "training": {
                "running": self.trainer.running,
                "starting": self.trainer.starting,
                "until": self.trainer.until_step,
                "startedAt": self.trainer.started_at_ms,
                "stopRequested": self.trainer.stop_requested,
                "outcome": self.trainer.outcome,
            },
            "casting": self.trainer.casting,
            "generating": False,
            "inference": self.trainer.inference_status(),
            "models": self.trainer.list_library_models(),
            "foraging": self.forager.status(),
            "teachers": {
                "running": self.teachers.running,
                "enabled": len(self.teachers.enabled_ids()),
            },
            "autosave": self.trainer.probe_autosave(),
            "github": {
                "repo": gh.get("repo", ""),
                "path": gh.get("path", "brains/ember.json"),
                "pullStart": bool(gh.get("pullStart", False)),
                "hasToken": bool(gh.get("token")),
            },
            "losses": self.trainer.losses[-400:],
            "validations": self.trainer.validations[-120:],
            "samples": self.trainer.samples[:8],
            "trainLog": self.trainer.train_log[-120:],
            "forageLog": self.forager.logs[-120:],
            "teacherLog": self.teachers.logs[-120:],
        }

    # ------------------------------------------------------------------
    # Export file helper (writes directly to cache/exports for zero-copy SAF save)
    # ------------------------------------------------------------------
    def _write_export_file(self, filename: str, payload_obj: Any, kind: str = "brain") -> dict[str, Any]:
        transfer_id = uuid.uuid4().hex[:12]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)[:120] or "ember-brain.json"
        out_path = self.exports_dir / f"{transfer_id}-{safe_name}"
        raw = payload_obj if isinstance(payload_obj, (bytes, bytearray)) else json.dumps(payload_obj, ensure_ascii=True).encode("utf-8")
        tmp = str(out_path) + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out_path)
        return {
            "transferId": transfer_id,
            "tempPath": str(out_path),
            "filename": safe_name,
            "bytes": len(raw),
            "kind": kind,
        }

    # ------------------------------------------------------------------
    # Method dispatch
    # ------------------------------------------------------------------
    def dispatch(self, method: str, params: dict[str, Any] | None = None) -> Any:
        p = params or {}

        if method == "ping":
            return {"pong": True, "ready": True, "at": int(time.time() * 1000)}

        if method == "runtime.info":
            engine = get_engine()
            return {
                "backend": "python + llama.cpp",
                "tried": ["python-numpy", "llama.cpp"],
                "runtime": f"Python {platform.python_version()}",
                "ready": True,
                "tfVersion": f"NumPy {np.__version__}",
                "inferenceBackend": "llama.cpp",
                "nativeAvailable": engine.available(),
                "nativeInfo": engine.backend_info(),
                "nativeError": engine.load_error,
                "modelFormat": EMBER_MODEL_FORMAT,
                "app": "ember-v3",
            }

        if method == "status":
            return self.full_status()

        # -- Corpus -----------------------------------------------------
        if method == "corpus.summary":
            return self.corpus.summary()

        if method == "corpus.list":
            return self.corpus.list_documents(
                query=str(p.get("query") or ""),
                offset=int(p.get("offset") or 0),
                limit=int(p.get("limit") or 40),
            )

        if method == "corpus.doc":
            doc = self.corpus.get_document(str(p.get("id") or ""))
            if doc is None:
                raise ValueError("Document no longer exists in corpus.")
            return doc

        if method == "corpus.addText":
            doc = self.corpus.add(
                text=str(p.get("text") or ""),
                title=str(p.get("title") or "untitled"),
                source=str(p.get("source") or "manual"),
                url=str(p.get("url") or ""),
                weight=int(p.get("weight") or 1),
                kind=str(p.get("kind") or "page"),
            )
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {
                "doc": doc,
                "document": doc,
                "corpus": summary,
                "merged": bool(doc.get("merged", False)),
            }

        if method == "corpus.addPack":
            pack_id = str(p.get("pack") or p.get("id") or "starter")
            docs = get_pack(pack_id)
            counts = self.corpus.add_many(docs)
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {
                "pack": pack_id,
                "title": PACK_TITLES.get(pack_id, f"{pack_id.capitalize()} pack"),
                "added": counts["accepted"],
                "merged": counts["merged"],
                "total": len(docs),
                "corpus": summary,
            }

        if method == "corpus.remove":
            removed = self.corpus.remove(str(p.get("id") or ""))
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {"removed": removed, "corpus": summary}

        if method == "corpus.clear":
            removed = self.corpus.clear()
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {"cleared": True, "removed": removed, "corpus": summary}

        if method == "corpus.dedupe":
            removed = self.corpus.dedupe()
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {"removed": removed, "corpus": summary}

        if method == "lesson.add":
            prompt_txt = str(p.get("prompt") or "").strip()
            resp_txt = str(p.get("response") or "").strip()
            if not resp_txt:
                raise ValueError("Write the target response you want Ember to learn.")
            weight = max(1, min(30, int(p.get("weight") or 8)))
            lesson_body = f"### Prompt:\n{prompt_txt}\n\n### Response:\n{resp_txt}\n"
            doc = self.corpus.add(
                text=lesson_body,
                title=(prompt_txt[:80] or "Hand-taught lesson"),
                source="taught",
                kind="lesson",
                weight=weight,
            )
            summary = self.corpus.summary()
            self.emit("corpus.changed", summary)
            return {"doc": doc, "document": doc, "corpus": summary}

        # -- Dataset & Brain File Ingestion -----------------------------
        if method == "file.inspectText":
            tok = str(p.get("token") or f"txt-{uuid.uuid4().hex[:8]}")
            return self.importer.stage_text(
                token=tok,
                text=str(p.get("text") or ""),
                name=str(p.get("name") or "pasted.txt"),
                mime=str(p.get("mime") or "text/plain"),
            )

        if method == "file.inspect":
            tok = str(p.get("token") or f"file-{uuid.uuid4().hex[:8]}")
            fpath = p.get("path")
            if not fpath:
                raise ValueError("Missing staged file path for inspection.")
            return self.importer.stage_path(
                token=tok,
                path=str(fpath),
                name=str(p.get("name") or ""),
                mime=str(p.get("mime") or ""),
            )

        if method in ("file.commit", "file.resume"):
            tok = str(p.get("token") or "")
            fn = self.importer.resume if method == "file.resume" else self.importer.commit
            res = fn(
                token=tok,
                path=p.get("path"),
                on_progress=lambda prog: self.emit("import.progress", prog),
            )
            self.emit("import.finished", {"token": tok, "outcome": res.get("outcome", "completed")})
            self.emit("corpus.changed", self.corpus.summary())
            return res

        if method == "file.pause":
            return self.importer.pause(str(p.get("token") or ""))

        if method == "file.cancel":
            return self.importer.cancel(str(p.get("token") or ""))

        if method == "file.discard":
            return self.importer.discard(str(p.get("token") or ""))

        if method == "file.imports":
            return {"imports": self.importer.list_imports()}

        # -- Vessel & Training ------------------------------------------
        if method == "vessel.estimate":
            return estimate_vessel(p)

        if method == "vessel.cast":
            return self.trainer.cast(p)

        if method == "train.start":
            return self.trainer.start_background(p)

        if method == "train.sync":
            return self.trainer.train_sync(p)

        if method == "train.stop":
            return self.trainer.stop()

        if method == "train.status":
            return {
                "training": {
                    "running": self.trainer.running,
                    "starting": self.trainer.starting,
                    "until": self.trainer.until_step,
                    "startedAt": self.trainer.started_at_ms,
                    "stopRequested": self.trainer.stop_requested,
                    "outcome": self.trainer.outcome,
                },
                "model": self.trainer.status(),
                "losses": self.trainer.losses[-400:],
                "validations": self.trainer.validations[-120:],
                "samples": self.trainer.samples[:8],
                "forageLog": self.forager.logs[-120:],
                "teacherLog": self.teachers.logs[-120:],
            }

        # -- Prompt preparation & generation ----------------------------
        if method == "gen.prepare":
            lang = str(p.get("language") or p.get("lang") or "Spanish")
            formatted = format_task_prompt(
                self.corpus,
                str(p.get("prompt") or ""),
                task=str(p.get("task") or "continue"),
                language=lang,
                subreddit=str(p.get("subreddit") or ""),
            )
            try:
                if self.trainer._imported_gguf is None:
                    self.trainer.sync_gguf()
                gguf_path = self.trainer.inference_model_path
                if gguf_path is None:
                    raise InferenceError("No GGUF model available. Cast a vessel or import GGUF.", code=-3, name="model-not-loaded")
                validate_gguf_file(gguf_path)
                # Android owns its JNI engine; release any training-sample model
                # before loading the same weights there, avoiding duplicate RAM.
                get_engine().unload()
                self.trainer._engine_model_path = ""
                model_path = str(gguf_path)
                model_error = None
            except InferenceError as exc:
                model_path = ""
                model_error = exc.to_dict()
            return {
                "formattedPrompt": formatted,
                "modelPath": model_path,
                "tokenizerPath": str(self.trainer.active_tokenizer_path),
                "modelFormat": EMBER_MODEL_FORMAT,
                "inference": self.trainer.inference_status(),
                "modelError": model_error,
            }

        if method == "gen.run":
            lang = str(p.get("language") or p.get("lang") or "Spanish")
            formatted = format_task_prompt(
                self.corpus,
                str(p.get("prompt") or ""),
                task=str(p.get("task") or "continue"),
                language=lang,
                subreddit=str(p.get("subreddit") or ""),
            )
            seed_str = str(p.get("seed") or "").strip()
            seed_val = int(seed_str) if seed_str.lstrip("-").isdigit() else None
            # Explicit zeros must be honoured: `p.get("maxNew") or 80` would turn a
            # deliberate "generate nothing" request into 80 tokens.
            max_new = p.get("maxNew")
            try:
                res = self.trainer.generate_sample(
                    prompt=formatted,
                max_new_tokens=max(0, int(max_new if max_new is not None else 80)),
                temperature=float(p.get("temperature") if p.get("temperature") is not None else 0.9),
                top_k=int(p.get("topK") if p.get("topK") is not None else 40),
                top_p=float(p.get("topP") if p.get("topP") is not None else 0.95),
                repetition_penalty=float(p.get("repetition") if p.get("repetition") is not None else 1.15),
                    seed=seed_val,
                    stop_on_eos=bool(p.get("stopOnBoundary", True)),
                )
            except InferenceError as exc:
                return {
                    "text": "",
                    "meta": "",
                    "newTokens": 0,
                    "tokensPerSec": 0.0,
                    "ms": 0.0,
                    "promptTokens": 0,
                    "seed": None,
                    "stoppedBy": "error",
                    "warning": None,
                    "ok": False,
                    **exc.to_dict(),
                }
            return {
                "ok": res.ok,
                "text": res.text,
                "meta": f"{res.new_tokens} tokens · {res.tokens_per_sec:.1f} tok/s · seed {res.seed}"
                        + (f" · {res.stopped_by}" if res.stopped_by not in ("length",) else ""),
                "newTokens": res.new_tokens,
                "tokenIds": res.token_ids,
                "tokensPerSec": res.tokens_per_sec,
                "ms": res.ms,
                "promptMs": res.prompt_ms,
                "genMs": res.gen_ms,
                "promptTokens": res.prompt_tokens,
                "seed": res.seed,
                "stoppedBy": res.stopped_by,
                "warning": res.warning,
                "nCtx": res.n_ctx,
                "error": res.error,
                "errorCode": res.error_code,
                "errorName": res.error_name,
                "backend": "llama.cpp",
            }

        if method == "gen.cancel":
            get_engine().cancel()
            return {"cancelled": True}

        # -- Native model management (llama.cpp / GGUF) -------------------
        if method == "model.info":
            engine = get_engine()
            payload: dict[str, Any] = {"inference": self.trainer.inference_status()}
            if engine.available() and engine.is_loaded():
                payload["model"] = engine.model_info().to_dict()
            if str(p.get("path") or ""):
                try:
                    payload["file"] = validate_gguf_file(str(p["path"]))
                except Exception as exc:
                    payload["file"] = {"valid": False, "error": str(exc)}
            return payload

        if method == "model.validate":
            target = str(p.get("path") or "") or str(self.trainer.inference_model_path or "")
            if not target:
                raise ValueError("No model file to validate: cast and fire, or import a .gguf model.")
            try:
                return validate_gguf_file(target)
            except Exception as exc:
                return {"path": target, "valid": False, "error": str(exc)}

        if method == "model.list":
            return {"models": self.trainer.list_library_models(), "format": EMBER_MODEL_FORMAT}

        if method == "model.select":
            raw = p.get("path")
            return self.trainer.select_model(None if raw in (None, "") else str(raw))

        if method == "model.importGguf":
            src = str(p.get("path") or "")
            if not src:
                raise ValueError("Missing staged model path for import.")
            return self.trainer.import_gguf(src, name=str(p.get("name") or ""), select=p.get("select") is not False)

        if method == "model.exportGguf":
            if not self.trainer.ready:
                raise ValueError("Nothing to export — cast a vessel and fire first.")
            quantization = str(p.get("quantization") or self.trainer.quantization or "f32")
            name = str(p.get("name") or f"ember-step-{self.trainer.step}").strip() or f"ember-step-{self.trainer.step}"
            if name.lower().endswith(".gguf"):
                name = name[:-5]
            gguf_path = self.trainer.sync_gguf(force=True, quantization=quantization)
            if gguf_path is None:
                raise ValueError("No trained vessel to export — cast and fire first.")
            payload = self._write_export_file(f"{name}.gguf", Path(gguf_path).read_bytes(), kind="model")
            payload["model"] = validate_gguf_file(gguf_path)
            payload["quantization"] = quantization
            return payload

        if method == "model.unload":
            return self.trainer.unload_engine_model()

        if method == "model.setQuantization":
            quantization = str(p.get("quantization") or "f32").strip().lower()
            if quantization not in ("f32", "q8_0"):
                raise ValueError("Quantization must be 'f32' or 'q8_0'.")
            self.trainer.quantization = quantization
            if self.trainer.ready:
                self.trainer.sync_gguf(force=True)
            return {"quantization": quantization, "inference": self.trainer.inference_status()}

        # -- Brain export/import & Checkpoints --------------------------
        if method == "brain.exportJson":
            name = str(p.get("name") or f"ember-step-{self.trainer.step}").strip()
            if name.lower().endswith(".json"):
                name = name[:-5]
            include_corpus = p.get("includeCorpus") is not False
            return self.trainer.serialize_brain(name=name, include_corpus=include_corpus)

        if method == "brain.export":
            raw_name = str(p.get("name") or f"ember-step-{self.trainer.step}").strip() or f"ember-step-{self.trainer.step}"
            if raw_name.lower().endswith(".emb"):
                mpath, _ = self.trainer._sync_active_files()
                return self._write_export_file(raw_name, mpath.read_bytes(), kind="brain")
            stem = raw_name[:-5] if raw_name.lower().endswith(".json") else raw_name
            include_corpus = p.get("includeCorpus") is not False
            brain = self.trainer.serialize_brain(name=stem, include_corpus=include_corpus)
            return self._write_export_file(f"{stem}.json", brain, kind="brain")

        if method == "brain.importText":
            raw_txt = str(p.get("text") or "")
            data = json.loads(raw_txt)
            return self.trainer.import_brain(
                data,
                name=str(p.get("name") or "imported"),
                save_checkpoint=p.get("saveCheckpoint") is not False,
            )

        if method == "brain.importFile":
            fpath = Path(str(p.get("path") or ""))
            if not fpath.exists():
                raise ValueError("Staged brain file does not exist.")
            raw_bytes = fpath.read_bytes()
            stem_name = str(p.get("name") or fpath.stem)
            if stem_name.lower().endswith(".emb") or stem_name.lower().endswith(".json"):
                stem_name = Path(stem_name).stem
            if raw_bytes[:4] == FORMAT_MAGIC:
                companion = fpath.with_suffix(".tok.json")
                return self.trainer.import_brain_emb(
                    raw_bytes,
                    name=stem_name,
                    companion_tok_path=companion if companion.exists() else None,
                    save_checkpoint=p.get("saveCheckpoint") is not False,
                )
            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            return self.trainer.import_brain(
                data,
                name=stem_name,
                save_checkpoint=p.get("saveCheckpoint") is not False,
            )

        if method == "ckpt.list":
            return {"checkpoints": self.trainer.list_checkpoints()}

        if method == "ckpt.save":
            info = self.trainer.save_checkpoint(str(p.get("name") or f"step-{self.trainer.step}"))
            return {**info, "checkpoints": self.trainer.list_checkpoints()}

        if method == "ckpt.load":
            return self.trainer.load_checkpoint(str(p.get("name") or "autosave"))

        if method == "ckpt.delete":
            removed = self.trainer.delete_checkpoint(str(p.get("name") or ""))
            return {"deleted": removed, "checkpoints": self.trainer.list_checkpoints()}

        if method == "ckpt.export":
            name = str(p.get("name") or "autosave")
            brain = self.trainer.export_checkpoint_dict(name, include_corpus=False)
            return self._write_export_file(f"checkpoint-{name}.json", brain, kind="checkpoint")

        # -- Forager & Teachers -----------------------------------------
        if method == "forage.start":
            return self.forager.start(p.get("options") if isinstance(p.get("options"), dict) else p)

        if method == "forage.stop":
            return self.forager.stop()

        if method == "teachers.list":
            return {
                "running": self.teachers.running,
                "windows": WINDOWS,
                "teachers": self.teachers.descriptors(),
            }

        if method == "teachers.update":
            updated = self.teachers.update(str(p.get("id") or ""), dict(p.get("fields") or {}))
            return {"teachers": updated}

        if method == "teachers.defaultWindow":
            updated = self.teachers.set_default_window(str(p.get("window") or "daily"))
            return {"teachers": updated}

        if method == "teachers.run":
            return self.teachers.run_auto(str(p.get("topics") or ""))

        if method == "teachers.stop":
            return self.teachers.stop_auto()

        if method == "teachers.ask":
            return self.teachers.ask(str(p.get("id") or ""), str(p.get("topic") or ""))

        if method == "teachers.acceptDraft":
            return self.teachers.accept_draft(
                str(p.get("id") or ""),
                str(p.get("topic") or ""),
                str(p.get("text") or ""),
            )

        # -- GitHub Sync ------------------------------------------------
        if method == "github.settings":
            gh = self.corpus.get_kv("github") or {}
            return {
                "repo": gh.get("repo", ""),
                "path": gh.get("path", "brains/ember.json"),
                "token": gh.get("token", ""),
                "pullStart": bool(gh.get("pullStart", False)),
            }

        if method == "github.save":
            gh = {
                "repo": str(p.get("repo") or "").strip(),
                "path": str(p.get("path") or "brains/ember.json").strip(),
                "token": str(p.get("token") or "").strip(),
                "pullStart": bool(p.get("pullStart", False)),
            }
            self.corpus.set_kv("github", gh)
            return {"saved": True}

        if method in ("github.pull", "github.push"):
            saved = self.corpus.get_kv("github") or {}
            repo_full = str(p.get("repo") or saved.get("repo") or "").strip()
            path = str(p.get("path") or saved.get("path") or "brains/ember.json").strip()
            token = str(p.get("token") or saved.get("token") or "").strip()
            parts = repo_full.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1] or not token:
                raise ValueError("Set GitHub owner/repo, path, and personal access token first.")
            owner, repo = parts[0], parts[1]
            if method == "github.pull":
                pulled = github_pull_brain(owner, repo, path, token)
                if not pulled["brain"]:
                    raise ValueError(f"No brain file found at {repo_full}/{path}.")
                res = self.trainer.import_brain(pulled["brain"], name="github", save_checkpoint=True)
                self.corpus.set_kv("github-sha", pulled["sha"])
                return {**res, "sha": pulled["sha"], "repo": repo_full, "path": path}
            else:
                brain = self.trainer.serialize_brain(
                    name=f"step-{self.trainer.step}",
                    include_corpus=p.get("includeCorpus") is not False,
                )
                head = github_head(owner, repo, path, token)
                sha = github_push_brain(owner, repo, path, token, brain, head.get("sha"))
                self.corpus.set_kv("github-sha", sha)
                return {"pushed": True, "sha": sha, "repo": repo_full, "path": path}

        raise ValueError(f"Unknown backend method: {method}")


# ==========================================================================
# Module-level singleton entry points used by Chaquopy on Android
# ==========================================================================

_INSTANCE_LOCK = threading.Lock()
_INSTANCE: EmberBackend | None = None


def init_android(
    data_dir: str,
    cache_dir: str,
    kotlin_bridge: Any = None,
) -> str:
    """Initialize process-wide Python backend instance from Kotlin (Chaquopy)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return json.dumps({"ok": True, "reused": True})

        def _on_event(event_name: str, payload: dict[str, Any]) -> None:
            if kotlin_bridge is not None:
                try:
                    kotlin_bridge.onPythonEvent(event_name, json.dumps(payload, ensure_ascii=False))
                except Exception:
                    pass

        def _on_sync(model_path: str, tok_path: str) -> None:
            if kotlin_bridge is not None:
                try:
                    kotlin_bridge.onModelSynced(model_path, tok_path)
                except Exception:
                    pass

        _INSTANCE = EmberBackend(
            data_dir=data_dir,
            cache_dir=cache_dir,
            on_event=_on_event,
            on_model_synced=_on_sync,
        )
        return json.dumps({"ok": True, "reused": False})


def rpc_json(request_text: str) -> str:
    """Execute one JSON-RPC call from Kotlin and return a JSON envelope."""
    global _INSTANCE
    if _INSTANCE is None:
        return json.dumps({"ok": False, "error": "Python backend has not been initialized yet."})
    try:
        req = json.loads(request_text)
        method = str(req.get("method") or "")
        params = req.get("params") if isinstance(req.get("params"), dict) else {}
        result = _INSTANCE.dispatch(method, params)
        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def shutdown_android() -> None:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            _INSTANCE.close()
            _INSTANCE = None
