"""Unified Ember v3 JSON-RPC backend (Android Chaquopy + Desktop HTTP + Tests)."""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corpus import Corpus
from .forager import Forager
from .ingest import ImportManager
from .packs import get_pack
from .teachers import WINDOWS, TeacherDesk, github_head, github_pull_brain, github_push_brain
from .trainer import Trainer, estimate_vessel


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
        lang = (language or "Spanish").strip()
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
        self.forager = Forager(self.corpus, emit_event=self.emit)
        self.teachers = TeacherDesk(self.corpus, emit_event=self.emit)
        self.autosave = self.trainer.probe_autosave()

        # Automatically restore active.emb if saved from a previous app session
        if self.trainer.active_model_path.exists() and self.trainer.active_tokenizer_path.exists():
            try:
                from .model import GPT, load_model
                from .optim import AdamW
                from .tokenizer import BPETokenizer

                cfg, params, meta = load_model(self.trainer.active_model_path)
                tok = BPETokenizer.load(self.trainer.active_tokenizer_path)
                self.trainer.cfg = cfg
                self.trainer.tokenizer = tok
                self.trainer.model = GPT(cfg, params=params)
                self.trainer.optimizer = AdamW(self.trainer.model.params, lr=3e-4)
                self.trainer.step = int(meta.get("step", 0))
                self.trainer.rebuild_streams_if_needed()
                self._handle_model_synced(self.trainer.active_model_path, self.trainer.active_tokenizer_path)
            except Exception:
                pass

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
        self.corpus.close()

    # ------------------------------------------------------------------
    # Full status snapshot
    # ------------------------------------------------------------------
    def full_status(self) -> dict[str, Any]:
        gh = self.corpus.get_kv("github") or {}
        return {
            "ready": True,
            "backend": "python-numpy + native-cpp",
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
        }

    # ------------------------------------------------------------------
    # Export file helper (writes directly to cache/exports for zero-copy SAF save)
    # ------------------------------------------------------------------
    def _write_export_file(self, filename: str, payload_obj: Any, kind: str = "brain") -> dict[str, Any]:
        transfer_id = uuid.uuid4().hex[:12]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)[:120] or "ember-brain.json"
        out_path = self.exports_dir / f"{transfer_id}-{safe_name}"
        raw = json.dumps(payload_obj, ensure_ascii=True).encode("utf-8")
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
            return {
                "backend": "python + c++",
                "tried": ["python", "native-cpp"],
                "runtime": f"Python {platform.python_version()}",
                "ready": True,
                "tfVersion": f"NumPy {np.__version__}",
                "nativeAvailable": self.trainer._native.available,
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
            self.emit("corpus.changed", self.corpus.summary())
            return {"doc": doc, "merged": bool(doc.get("merged", False))}

        if method == "corpus.addPack":
            docs = get_pack(str(p.get("pack") or p.get("id") or "starter"))
            counts = self.corpus.add_many(docs)
            self.emit("corpus.changed", self.corpus.summary())
            return {"added": counts["accepted"], "merged": counts["merged"], "total": len(docs)}

        if method == "corpus.remove":
            removed = self.corpus.remove(str(p.get("id") or ""))
            self.emit("corpus.changed", self.corpus.summary())
            return {"removed": removed}

        if method == "corpus.clear":
            removed = self.corpus.clear()
            self.emit("corpus.changed", self.corpus.summary())
            return {"cleared": True, "removed": removed}

        if method == "corpus.dedupe":
            removed = self.corpus.dedupe()
            self.emit("corpus.changed", self.corpus.summary())
            return {"removed": removed}

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
            self.emit("corpus.changed", self.corpus.summary())
            return {"doc": doc}

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
            }

        # -- Prompt preparation & generation ----------------------------
        if method == "gen.prepare":
            formatted = format_task_prompt(
                self.corpus,
                str(p.get("prompt") or ""),
                task=str(p.get("task") or "continue"),
                language=str(p.get("language") or "Spanish"),
                subreddit=str(p.get("subreddit") or ""),
            )
            mpath, tpath = self.trainer._sync_active_files()
            return {
                "formattedPrompt": formatted,
                "modelPath": str(mpath),
                "tokenizerPath": str(tpath),
            }

        if method == "gen.run":
            formatted = format_task_prompt(
                self.corpus,
                str(p.get("prompt") or ""),
                task=str(p.get("task") or "continue"),
                language=str(p.get("language") or "Spanish"),
                subreddit=str(p.get("subreddit") or ""),
            )
            seed_str = str(p.get("seed") or "").strip()
            seed_val = int(seed_str) if seed_str.lstrip("-").isdigit() else None
            res = self.trainer.generate_sample(
                prompt=formatted,
                max_new_tokens=int(p.get("maxNew") or 80),
                temperature=float(p.get("temperature") if p.get("temperature") is not None else 0.9),
                top_k=int(p.get("topK") if p.get("topK") is not None else 40),
                top_p=float(p.get("topP") if p.get("topP") is not None else 0.95),
                repetition_penalty=float(p.get("repetition") if p.get("repetition") is not None else 1.15),
                seed=seed_val,
                stop_on_eos=bool(p.get("stopOnBoundary", True)),
            )
            return {
                "text": res.text,
                "meta": f"{res.new_tokens} tokens · {res.tokens_per_sec:.1f} tok/s · seed {res.seed}",
                "newTokens": res.new_tokens,
                "tokensPerSec": res.tokens_per_sec,
                "ms": res.ms,
                "promptTokens": res.prompt_tokens,
                "seed": res.seed,
                "warning": res.warning,
            }

        # -- Brain export/import & Checkpoints --------------------------
        if method == "brain.exportJson":
            name = str(p.get("name") or f"ember-step-{self.trainer.step}")
            include_corpus = p.get("includeCorpus") is not False
            return self.trainer.serialize_brain(name=name, include_corpus=include_corpus)

        if method == "brain.export":
            name = str(p.get("name") or f"ember-step-{self.trainer.step}")
            include_corpus = p.get("includeCorpus") is not False
            brain = self.trainer.serialize_brain(name=name, include_corpus=include_corpus)
            return self._write_export_file(f"{name}.json", brain, kind="brain")

        if method == "brain.importText":
            data = json.loads(str(p.get("text") or ""))
            return self.trainer.import_brain(
                data,
                name=str(p.get("name") or "imported"),
                save_checkpoint=p.get("saveCheckpoint") is not False,
            )

        if method == "brain.importFile":
            fpath = Path(str(p.get("path") or ""))
            if not fpath.exists():
                raise ValueError("Staged brain file does not exist.")
            data = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
            return self.trainer.import_brain(
                data,
                name=str(p.get("name") or fpath.stem),
                save_checkpoint=p.get("saveCheckpoint") is not False,
            )

        if method == "ckpt.list":
            return {"checkpoints": self.trainer.list_checkpoints()}

        if method == "ckpt.save":
            return self.trainer.save_checkpoint(str(p.get("name") or f"step-{self.trainer.step}"))

        if method == "ckpt.load":
            return self.trainer.load_checkpoint(str(p.get("name") or "autosave"))

        if method == "ckpt.delete":
            return {"deleted": self.trainer.delete_checkpoint(str(p.get("name") or ""))}

        if method == "ckpt.export":
            name = str(p.get("name") or "autosave")
            self.trainer.load_checkpoint(name)
            brain = self.trainer.serialize_brain(name=name, include_corpus=False)
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
