"""SQLite-backed corpus, key-value store, deduplication, health & retrieval.

Designed for mobile memory bounds:
- Documents stay on disk in SQLite (WAL mode) rather than duplicated in RAM.
- Iteration streams batches via SQL cursors (`iter_documents`).
- Summary and corpus health run directly from SQL/metadata without loading
  full texts on large corpora.
- Existing legacy `.jsonl` corpus files are automatically imported once so user
  training material is preserved.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from .dataset import Document, normalize_split, compute_holdout_ids

MAX_DOC_CHARS = 120_000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_NL_RE = re.compile(r"\n{4,}")
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def clean_document_text(raw: str, strip_html: bool = False) -> str:
    """Normalize text, remove control junk, decode HTML entities when present."""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_RE.sub("", text)
    if strip_html or ("</" in text and ("<script" in text.lower() or "<style" in text.lower() or "<p>" in text.lower())):
        text = _SCRIPT_STYLE_RE.sub(" ", text)
        if strip_html:
            text = _TAG_RE.sub(" ", text)
        text = html.unescape(text)
    elif "&amp;" in text or "&quot;" in text or "&#" in text or "&lt;" in text:
        text = html.unescape(text)
    text = _MULTI_NL_RE.sub("\n\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    norm = " ".join(text.split()).lower()
    return hashlib.blake2b(norm.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


class Corpus:
    """Persistent thread-safe corpus backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        raw_path = Path(path)
        self._lock = threading.RLock()
        self._revision = 0
        if raw_path.suffix == ".jsonl":
            self.jsonl_path: Path | None = raw_path
            self.db_path = raw_path.with_suffix(".sqlite3")
        elif raw_path.suffix in (".db", ".sqlite", ".sqlite3"):
            self.jsonl_path = None
            self.db_path = raw_path
        else:
            raw_path.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = raw_path / "corpus.jsonl"
            self.db_path = raw_path / "ember.sqlite3"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        if self.jsonl_path is not None and self.jsonl_path.exists():
            self._import_legacy_jsonl(self.jsonl_path)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS documents (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    url TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'page',
                    weight INTEGER NOT NULL DEFAULT 1,
                    chars INTEGER NOT NULL DEFAULT 0,
                    chash TEXT NOT NULL DEFAULT '',
                    added_at REAL NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    text TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_docs_chash ON documents(chash);
                CREATE INDEX IF NOT EXISTS idx_docs_url ON documents(url) WHERE url != '';
                CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source);

                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()

    def _import_legacy_jsonl(self, jsonl_path: Path) -> None:
        """Preserve pre-existing JSONL corpus documents on first open."""
        marker = f"migrated:{jsonl_path.name}:{jsonl_path.stat().st_size}"
        if self.get_kv("legacy_jsonl_marker") == marker:
            return
        existing = self.count()
        if existing == 0:
            batch: list[dict[str, Any]] = []
            try:
                with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and obj.get("text"):
                            batch.append(obj)
                        if len(batch) >= 200:
                            self.add_many(batch)
                            batch.clear()
                if batch:
                    self.add_many(batch)
            except OSError:
                pass
        self.set_kv("legacy_jsonl_marker", marker)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # -- KV store -------------------------------------------------------
    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value_json FROM kv WHERE key = ?", (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value_json"])
            except json.JSONDecodeError:
                return default

    def set_kv(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(key, value_json, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def delete_kv(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.commit()

    # -- CRUD & batch insertion -----------------------------------------
    @staticmethod
    def _merge_split_metadata(old,new):
        old=old if isinstance(old,dict) else {};new=new if isinstance(new,dict) else {}
        result={**old,**new};rank={"":0,"train":1,"validation":2,"test":3}
        protected=max((normalize_split(old.get("split")),normalize_split(new.get("split"))),key=lambda s:rank[s])
        if protected:result["split"]=protected
        return result

    def _protect_content_split(self,cursor,chash,meta):
        rows=cursor.execute("SELECT id,meta_json FROM documents WHERE chash=?",(chash,)).fetchall()
        protected=meta
        for r in rows:protected=self._merge_split_metadata(json.loads(r["meta_json"]),protected)
        split=normalize_split(protected.get("split"))
        if split:
            for r in rows:
                old=json.loads(r["meta_json"])
                if normalize_split(old.get("split"))!=split:
                    cursor.execute("UPDATE documents SET meta_json=? WHERE id=?",(json.dumps({**old,"split":split}),r["id"]))
        return protected

    def add(
        self,
        text: str,
        title: str = "untitled",
        source: str = "manual",
        url: str = "",
        weight: int = 1,
        kind: str = "page",
        meta: dict[str, Any] | None = None,
        strip_html: bool = False,
        dedupe: bool = True,
    ) -> dict[str, Any]:
        cleaned = clean_document_text(text, strip_html=strip_html)
        if not cleaned:
            raise ValueError("Document text is empty after cleaning.")

        safe_title = (str(title or "").strip()[:240]) or "untitled"
        safe_source = (str(source or "manual").strip()[:80]) or "manual"
        safe_url = str(url or "").strip()[:1024]
        safe_kind = "lesson" if kind == "lesson" else "page"
        safe_weight = max(1, min(50, int(round(float(weight or 1)))))
        chash = content_hash(cleaned)
        meta_str = json.dumps(meta or {}, ensure_ascii=False)
        now = time.time()

        with self._lock:
            if safe_url:
                row = self._conn.execute(
                    "SELECT id,meta_json FROM documents WHERE url = ? LIMIT 1", (safe_url,)
                ).fetchone()
                if row is not None:
                    merged_meta=self._merge_split_metadata(json.loads(row["meta_json"]),meta or {})
                    meta_str=json.dumps(self._protect_content_split(self._conn,chash,merged_meta))
                    self._conn.execute(
                        "UPDATE documents SET title=?, text=?, chars=?, chash=?, weight=?, meta_json=? WHERE id=?",
                        (safe_title, cleaned, len(cleaned), chash, safe_weight, meta_str, row["id"]),
                    )
                    self._conn.commit()
                    self._revision += 1
                    return self.get_document(row["id"]) or {}

            meta_str=json.dumps(self._protect_content_split(self._conn,chash,meta or {}))
            if dedupe:
                row = self._conn.execute(
                    "SELECT id, weight FROM documents WHERE chash = ? LIMIT 1", (chash,)
                ).fetchone()
                if row is not None:
                    new_w = max(int(row["weight"]), safe_weight)
                    self._conn.execute(
                        "UPDATE documents SET weight=? WHERE id=?",
                        (new_w, row["id"]),
                    )
                    self._conn.commit()
                    self._revision += 1
                    doc = self.get_document(row["id"]) or {}
                    doc["merged"] = True
                    return doc

            # Derive the id from the content hash instead of a random UUID: the
            # train/validation holdout is a hash of the document id, so random ids
            # reshuffled the split on every import and made runs irreproducible
            # even with identical corpus text and a fixed model seed.
            doc_id = chash[:12]
            self._conn.execute(
                """
                INSERT INTO documents(id, title, source, url, kind, weight, chars, chash, added_at, meta_json, text)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    safe_title,
                    safe_source,
                    safe_url,
                    safe_kind,
                    safe_weight,
                    len(cleaned),
                    chash,
                    now,
                    meta_str,
                    cleaned,
                ),
            )
            self._conn.commit()
            self._revision += 1
            doc = self.get_document(doc_id) or {}
            doc["merged"] = False
            return doc

    def add_many(self, items: list[dict[str, Any]], dedupe: bool = True) -> dict[str, int]:
        """Insert a batch inside a single SQLite transaction for fast imports."""
        accepted = 0
        merged = 0
        skipped = 0
        duplicates = 0
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                for raw in items:
                    cleaned = clean_document_text(str(raw.get("text") or ""))
                    if not cleaned:
                        skipped += 1
                        continue
                    title = (str(raw.get("title") or "").strip()[:240]) or cleaned[:60].splitlines()[0][:60] or "untitled"
                    source = (str(raw.get("source") or "import").strip()[:80]) or "import"
                    url = str(raw.get("url") or "").strip()[:1024]
                    kind = "lesson" if raw.get("kind") == "lesson" else "page"
                    weight = max(1, min(50, int(round(float(raw.get("weight") or 1)))))
                    chash = content_hash(cleaned)
                    meta_str = json.dumps(raw.get("meta") or {}, ensure_ascii=False)

                    if url:
                        row = cur.execute("SELECT id,meta_json FROM documents WHERE url = ? LIMIT 1", (url,)).fetchone()
                        if row is not None:
                            merged_meta=self._merge_split_metadata(json.loads(row["meta_json"]),raw.get("meta") or {})
                            meta_str=json.dumps(self._protect_content_split(cur,chash,merged_meta))
                            cur.execute(
                                "UPDATE documents SET title=?, text=?, chars=?, chash=?, weight=?, meta_json=? WHERE id=?",
                                (title, cleaned, len(cleaned), chash, weight, meta_str, row["id"]),
                            )
                            merged += 1
                            continue

                    meta_str=json.dumps(self._protect_content_split(cur,chash,raw.get("meta") or {}))
                    if dedupe:
                        row = cur.execute("SELECT id FROM documents WHERE chash = ? LIMIT 1", (chash,)).fetchone()
                        if row is not None:
                            merged += 1
                            duplicates += 1
                            continue

                    doc_id = str(raw.get("id") or chash[:12])
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO documents(id, title, source, url, kind, weight, chars, chash, added_at, meta_json, text)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (doc_id, title, source, url, kind, weight, len(cleaned), chash, now, meta_str, cleaned),
                    )
                    accepted += 1
                self._conn.commit()
                if accepted or merged:
                    self._revision += 1
            except Exception:
                self._conn.rollback()
                raise
        return {"accepted": accepted, "merged": merged, "skipped": skipped, "duplicates": duplicates}

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
            if cur.rowcount > 0:
                self._revision += 1
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            n = self.count()
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()
            if n > 0:
                self._revision += 1
            return n

    def dedupe(self) -> int:
        """Remove exact content duplicates, keeping the earliest row and max weight."""
        with self._lock:
            before = self.count()
            self._conn.executescript(
                """
                UPDATE documents
                SET weight = (
                    SELECT MAX(d2.weight) FROM documents d2 WHERE d2.chash = documents.chash
                )
                WHERE seq IN (SELECT MIN(seq) FROM documents GROUP BY chash);

                DELETE FROM documents
                WHERE seq NOT IN (SELECT MIN(seq) FROM documents GROUP BY chash);
                """
            )
            self._conn.commit()
            removed = before - self.count()
            if removed > 0:
                self._revision += 1
            return removed

    # -- Queries & streaming iteration ----------------------------------
    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
            return int(row["n"]) if row else 0

    def id_kind_pairs(self) -> list[tuple[str, str]]:
        """Lightweight metadata-only fetch for deterministic train/val partitioning."""
        with self._lock:
            rows = self._conn.execute("SELECT id, kind FROM documents ORDER BY seq ASC").fetchall()
            return [(str(r["id"]), str(r["kind"])) for r in rows]

    def training_holdout_ids(self,frac=.05):
        # Only lightweight metadata, never document bodies. Content groups share
        # the most protected explicit split, or one deterministic auto split.
        with self._lock:
            rows=self._conn.execute("SELECT id,kind,chash,meta_json FROM documents ORDER BY seq").fetchall()
        explicit={}
        rank={"":0,"train":1,"validation":2,"test":3}
        for r in rows:
            split=normalize_split(json.loads(r["meta_json"]).get("split"))
            if rank[split]>rank.get(explicit.get(r["chash"],""),0):explicit[r["chash"]]=split
        auto=compute_holdout_ids([(r["id"],r["kind"]) for r in rows if r["chash"] not in explicit],frac)
        val_hashes={r["chash"] for r in rows if r["id"] in auto}
        return {r["id"] for r in rows if explicit.get(r["chash"]) in ("validation","test") or (r["chash"] not in explicit and r["chash"] in val_hashes)}

    def iter_documents(self, batch_size: int = 64) -> Iterator[Document]:
        """Stream :class:`Document` objects in bounded SQL pages without holding all in RAM."""
        last_seq = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT seq, id, title, source, kind, weight, text, meta_json
                    FROM documents
                    WHERE seq > ?
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (last_seq, batch_size),
                ).fetchall()
            if not rows:
                break
            for r in rows:
                last_seq = int(r["seq"])
                yield Document(
                    id=str(r["id"]),
                    title=str(r["title"]),
                    source=str(r["source"]),
                    kind=str(r["kind"]),
                    weight=float(r["weight"]),
                    text=str(r["text"]),
                    split=normalize_split(json.loads(r["meta_json"]).get("split")),
                )

    def iter_texts(self, batch_size: int = 64) -> Iterator[str]:
        for doc in self.iter_documents(batch_size=batch_size):
            if doc.text:
                yield doc.text

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if r is None:
                return None
            return self._row_to_dict(r, include_text=True)

    def list_documents(self, query: str = "", offset: int = 0, limit: int = 40) -> dict[str, Any]:
        q = (query or "").strip()
        offset = max(0, int(offset))
        limit = max(1, min(200, int(limit)))
        with self._lock:
            if q:
                pat = f"%{q}%"
                total = int(
                    self._conn.execute(
                        "SELECT COUNT(*) AS n FROM documents WHERE title LIKE ? OR source LIKE ? OR text LIKE ?",
                        (pat, pat, pat),
                    ).fetchone()["n"]
                )
                rows = self._conn.execute(
                    """
                    SELECT id, title, source, url, kind, weight, chars, added_at, meta_json,
                           SUBSTR(text, 1, 240) AS excerpt
                    FROM documents
                    WHERE title LIKE ? OR source LIKE ? OR text LIKE ?
                    ORDER BY seq DESC
                    LIMIT ? OFFSET ?
                    """,
                    (pat, pat, pat, limit, offset),
                ).fetchall()
            else:
                total = self.count()
                rows = self._conn.execute(
                    """
                    SELECT id, title, source, url, kind, weight, chars, added_at, meta_json,
                           SUBSTR(text, 1, 240) AS excerpt
                    FROM documents
                    ORDER BY seq DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

        items = [self._row_to_dict(r, include_text=False) for r in rows]
        return {
            "total": total,
            "offset": offset,
            "hasMore": (offset + len(items)) < total,
            "items": items,
        }

    def retrieve_passages(self, question: str, k: int = 3, max_chars: int = 1200) -> list[str]:
        """Lightweight lexical BM25-style retrieval over local documents for grounded QA."""
        terms = [t.lower() for t in _WORD_RE.findall(question or "")]
        if not terms:
            return []
        scored: list[tuple[float, str]] = []
        for doc in self.iter_documents(batch_size=128):
            if doc.kind == "lesson":
                continue
            low = doc.text.lower()
            hits = sum(1.0 / (1.0 + low.count(t) * 0.0) for t in set(terms) if t in low)
            if hits > 0:
                score = hits / (1.0 + len(doc.text) / 4000.0)
                scored.append((score, f"[{doc.title}]\n{doc.text[:max_chars]}"))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:k]]

    def summary(self, tokens_per_char: float = 0.28) -> dict[str, Any]:
        """Compute full corpus summary + health from metadata without materializing all texts."""
        with self._lock:
            agg = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(chars), 0) AS total_chars FROM documents"
            ).fetchone()
            n_docs = int(agg["n"])
            total_chars = int(agg["total_chars"])
            if n_docs == 0:
                return {
                    "documents": 0,
                    "characters": 0,
                    "bySource": {},
                    "by_source": {},
                    "health": {
                        "documents": 0,
                        "characters": 0,
                        "estTokens": 0,
                        "medianChars": 0,
                        "p25Chars": 0,
                        "p75Chars": 0,
                        "maxChars": 0,
                        "tinyDocs": 0,
                        "hugeDocs": 0,
                        "exactDuplicateDocs": 0,
                        "exactGroups": [],
                        "nearDuplicatePairs": 0,
                        "nonAsciiRatio": 0.0,
                        "issues": [],
                    },
                }

            by_source = {
                str(r["source"]): int(r["cnt"])
                for r in self._conn.execute(
                    "SELECT source, COUNT(*) AS cnt FROM documents GROUP BY source"
                ).fetchall()
            }
            lengths = [
                int(r[0])
                for r in self._conn.execute("SELECT chars FROM documents ORDER BY chars ASC").fetchall()
            ]
            dup_row = self._conn.execute(
                "SELECT COALESCE(SUM(cnt - 1), 0) AS dups FROM (SELECT COUNT(*) AS cnt FROM documents GROUP BY chash HAVING cnt > 1)"
            ).fetchone()
            exact_dups = int(dup_row["dups"]) if dup_row else 0

            sample_rows = self._conn.execute(
                "SELECT SUBSTR(text, 1, 1500) AS s FROM documents LIMIT 40"
            ).fetchall()

        sampled_chars = 0
        non_ascii = 0
        for sr in sample_rows:
            s = sr["s"] or ""
            sampled_chars += len(s)
            non_ascii += sum(1 for ch in s if ord(ch) > 127)

        def pct(p: float) -> int:
            idx = min(len(lengths) - 1, max(0, int(p * len(lengths))))
            return lengths[idx]

        tiny = sum(1 for c in lengths if c < 200)
        huge = sum(1 for c in lengths if c > 40_000)
        issues: list[str] = []
        if exact_dups:
            issues.append(f"{exact_dups} exact duplicate document{'s' if exact_dups != 1 else ''}")
        if tiny:
            issues.append(f"{tiny} tiny document{'s' if tiny != 1 else ''} (<200 chars)")
        if huge:
            issues.append(f"{huge} very long document{'s' if huge != 1 else ''} (>40k chars)")

        return {
            "documents": n_docs,
            "characters": total_chars,
            "bySource": by_source,
            "by_source": by_source,
            "health": {
                "documents": n_docs,
                "characters": total_chars,
                "estTokens": int(round(total_chars * tokens_per_char)),
                "medianChars": pct(0.5),
                "p25Chars": pct(0.25),
                "p75Chars": pct(0.75),
                "maxChars": lengths[-1],
                "tinyDocs": tiny,
                "hugeDocs": huge,
                "exactDuplicateDocs": exact_dups,
                "exactGroups": [],
                "nearDuplicatePairs": 0,
                "nonAsciiRatio": round(non_ascii / sampled_chars, 3) if sampled_chars else 0.0,
                "issues": issues,
            },
        }

    # Compatibility helpers used by older callers/tests
    def stats(self) -> dict[str, Any]:
        s = self.summary()
        page = self.list_documents(limit=80)
        return {
            "documents": s["documents"],
            "characters": s["characters"],
            "by_source": s["by_source"],
            "documents_preview": page["items"],
        }

    def texts(self) -> list[str]:
        return list(self.iter_texts())

    def export_docs(self, max_docs: int = 5000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, doc in enumerate(self.iter_documents()):
            if i >= max_docs:
                break
            out.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "kind": doc.kind,
                    "weight": int(round(doc.weight)),
                    "text": doc.text,
                }
            )
        return out

    @staticmethod
    def _row_to_dict(r: sqlite3.Row, include_text: bool) -> dict[str, Any]:
        keys = r.keys()
        meta = {}
        if "meta_json" in keys and r["meta_json"]:
            try:
                meta = json.loads(r["meta_json"])
            except json.JSONDecodeError:
                meta = {}
        text = str(r["text"]) if include_text and "text" in keys else ""
        excerpt = str(r["excerpt"]) if "excerpt" in keys else text[:240]
        added_ms = int(float(r["added_at"]) * 1000.0)
        d: dict[str, Any] = {
            "id": str(r["id"]),
            "title": str(r["title"]),
            "source": str(r["source"]),
            "url": str(r["url"]),
            "kind": str(r["kind"]),
            "weight": int(r["weight"]),
            "chars": int(r["chars"]),
            "addedAt": added_ms,
            "meta": meta,
            "excerpt": excerpt,
        }
        if include_text:
            d["text"] = text
        return d
