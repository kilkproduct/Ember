"""In-memory + JSONL corpus of scraped and hand-taught documents."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class Corpus:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.docs: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        docs: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.docs = docs

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def add(
        self,
        text: str,
        title: str = "untitled",
        source: str = "manual",
        url: str = "",
        weight: int = 1,
        kind: str = "page",
    ) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty document")
        # keep individual pages small enough to train on CPU
        if len(text) > 80_000:
            text = text[:80_000]
        doc = {
            "id": uuid.uuid4().hex[:12],
            "title": title[:240] or "untitled",
            "source": source,
            "url": url,
            "text": text,
            "weight": max(1, int(weight)),
            "kind": kind,
            "chars": len(text),
            "added_at": time.time(),
        }
        with self._lock:
            # de-dupe by url or identical title+prefix
            for existing in self.docs:
                if url and existing.get("url") == url:
                    existing.update({k: doc[k] for k in ("text", "chars", "weight", "title")})
                    self._rewrite()
                    return existing
            self.docs.append(doc)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return doc

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            before = len(self.docs)
            self.docs = [d for d in self.docs if d["id"] != doc_id]
            if len(self.docs) == before:
                return False
            self._rewrite()
            return True

    def clear(self) -> None:
        with self._lock:
            self.docs = []
            self._rewrite()

    def texts(self) -> list[str]:
        with self._lock:
            return [d["text"] for d in self.docs]

    def weighted_blob(self) -> str:
        """Concatenate documents, repeating higher-weight (taught) ones."""
        parts: list[str] = []
        with self._lock:
            for d in self.docs:
                chunk = d["text"].strip()
                if not chunk:
                    continue
                for _ in range(max(1, int(d.get("weight", 1)))):
                    parts.append(chunk)
        return "\n\n".join(parts)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            chars = sum(d.get("chars") or len(d.get("text", "")) for d in self.docs)
            by_source: dict[str, int] = {}
            for d in self.docs:
                src = d.get("source") or "unknown"
                by_source[src] = by_source.get(src, 0) + 1
            preview = [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "source": d["source"],
                    "url": d.get("url", ""),
                    "chars": d.get("chars") or len(d.get("text", "")),
                    "weight": d.get("weight", 1),
                    "kind": d.get("kind", "page"),
                    "excerpt": d.get("text", "")[:220],
                }
                for d in reversed(self.docs[-80:])
            ]
            return {
                "documents": len(self.docs),
                "characters": chars,
                "by_source": by_source,
                "documents_preview": preview,
            }
