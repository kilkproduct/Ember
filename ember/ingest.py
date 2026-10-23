"""Streaming multi-format dataset & brain inspector/importer for Ember.

Detects containers and formats from file bytes (never filename alone):
- Compression: gzip, bzip2, xz/lzma, zstd
- Archives: zip, tar (with zip-bomb ratio and path-traversal guards)
- Formats: JSONL/NDJSON, JSON array/object, CSV, TSV, Parquet, HTML, XML/RSS/Atom,
  Markdown, UTF-8 / UTF-8-BOM / UTF-16LE / UTF-16BE plain text, and Ember brains.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import html
import io
import json
import lzma
import os
import posixpath
import re
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ._parquet import decompress_zstd, iter_parquet_rows
from .corpus import Corpus, clean_document_text
from .model import BRAIN_KIND, BRAIN_VERSION, FORMAT_MAGIC

MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_RATIO = 80
MAX_ARCHIVE_ENTRIES = 5000
MAX_NESTED_DEPTH = 3

TEXT_KEYS = ("text", "body", "content", "article", "summary", "abstract", "passage", "markdown", "response")
TITLE_KEYS = ("title", "headline", "subject", "name", "heading", "id")
URL_KEYS = ("url", "link", "href", "uri")


def _safe_archive_member(name: str) -> str:
    """Reject traversal paths (`../`, leading `/`, Windows drives)."""
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned):
        raise ValueError(f"Unsafe archive entry path rejected: {name}")
    parts = [p for p in cleaned.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise ValueError(f"Archive path traversal rejected: {name}")
    norm = posixpath.normpath("/".join(parts))
    if norm.startswith(".."):
        raise ValueError(f"Archive path traversal rejected: {name}")
    return norm


def _decode_text_bytes(raw: bytes) -> tuple[str, str]:
    """Return ``(decoded_text, encoding_name)``."""
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace"), "utf-16le"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace"), "utf-16be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8-bom"
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _detect_compression(head: bytes) -> str | None:
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"BZh"):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    return None


def _decompress_bytes(raw: bytes, codec: str, max_bytes: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    src_len = max(1, len(raw))

    def _read_bounded(stream: io.BufferedIOBase) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            buf = stream.read(65536)
            if not buf:
                break
            total += len(buf)
            if total > max_bytes or (src_len > 64 and total > src_len * MAX_ARCHIVE_RATIO and total > 8 * 1024 * 1024):
                raise ValueError("Decompressed size or compression ratio exceeds safety limits.")
            chunks.append(buf)
        return b"".join(chunks)

    try:
        if codec == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                return _read_bounded(gz)  # type: ignore[arg-type]
        if codec == "bzip2":
            decomp = bz2.BZ2Decompressor()
            out = decomp.decompress(raw, max_length=max_bytes + 1)
            if not decomp.eof:
                raise ValueError("Truncated or oversized bzip2 stream.")
            return out
        if codec == "xz":
            decomp_xz = lzma.LZMADecompressor()
            out = decomp_xz.decompress(raw, max_length=max_bytes + 1)
            if not decomp_xz.eof:
                raise ValueError("Truncated or oversized xz stream.")
            return out
        if codec == "zstd":
            return decompress_zstd(raw, max_bytes=max_bytes)
    except (EOFError, OSError, lzma.LZMAError) as exc:
        raise ValueError(f"Corrupt or truncated {codec} stream: {exc}") from exc
    raise ValueError(f"Unsupported compression {codec}")


def _is_tar_bytes(raw: bytes) -> bool:
    return len(raw) >= 265 and raw[257:262] == b"ustar"


def _sniff_inner_format(raw: bytes, hint_name: str = "") -> tuple[str, str]:
    """Return ``(format_name, encoding)`` from decompressed payload bytes."""
    if raw.startswith(b"PAR1"):
        return "parquet", "binary"
    if raw.startswith(FORMAT_MAGIC):
        return "ember-model", "binary"

    sample, enc = _decode_text_bytes(raw[:65536])
    stripped = sample.lstrip()
    low_name = (hint_name or "").lower()

    if stripped.startswith("{") or stripped.startswith("["):
        # Distinguish multi-line NDJSON from single JSON document/array
        lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        if stripped.startswith("{") and len(lines) >= 2 and lines[0].endswith("}") and lines[1].startswith("{"):
            return "jsonl", enc
        try:
            full_text, _ = _decode_text_bytes(raw)
            obj = json.loads(full_text)
            if isinstance(obj, dict) and (
                obj.get("format") in ("ember-brain", "ember-brain-v1", "ember-brain-v2", "ember-brain-v3")
                or obj.get("kind") == BRAIN_KIND
                or ("weights" in obj and ("config" in obj or "cfg" in obj))
            ):
                return "brain", enc
            return "json", enc
        except json.JSONDecodeError:
            if stripped.startswith("{"):
                return "jsonl", enc
            return "json", enc

    if stripped.lower().startswith("<!doctype html") or re.search(r"<(html|body|article|main|p|h1)\b", stripped[:4000], re.I):
        return "html", enc
    if stripped.startswith("<?xml") or re.match(r"<(rss|feed|urlset|wikimedia|mediawiki)\b", stripped[:2000], re.I):
        return "xml", enc

    first_lines = [ln for ln in sample.splitlines()[:10] if ln.strip()]
    if len(first_lines) >= 2:
        if "\t" in first_lines[0] and first_lines[0].count("\t") == first_lines[1].count("\t"):
            return "tsv", enc
        if low_name.endswith(".csv") or (
            "," in first_lines[0]
            and first_lines[0].count(",") >= 1
            and first_lines[0].count(",") == first_lines[1].count(",")
            and len(first_lines[0]) < 400
        ):
            return "csv", enc

    if low_name.endswith((".md", ".markdown")) or re.search(r"(^|\n)#{1,3}\s+\S+", sample[:4000]):
        return "markdown", enc
    return "text", enc


def _extract_record_doc(obj: dict[str, Any], default_title: str, default_source: str) -> dict[str, Any] | None:
    low_map = {str(k).lower(): v for k, v in obj.items()}
    text_val = None
    for k in TEXT_KEYS:
        v = low_map.get(k)
        if isinstance(v, str) and v.strip():
            text_val = v
            break
    if text_val is None and "prompt" in low_map and ("response" in low_map or "completion" in low_map or "output" in low_map):
        ans = low_map.get("response") or low_map.get("completion") or low_map.get("output") or ""
        text_val = f"### Prompt:\n{str(low_map['prompt']).strip()}\n\n### Response:\n{str(ans).strip()}"
        kind = "lesson"
    else:
        kind = "lesson" if low_map.get("kind") == "lesson" else "page"

    if text_val is None:
        # Concatenate longest string fields if any
        strs = [str(v).strip() for k, v in obj.items() if isinstance(v, str) and len(v.strip()) >= 20 and k.lower() not in TITLE_KEYS]
        if strs:
            text_val = "\n\n".join(strs)
    if not text_val or not text_val.strip():
        return None

    title_val = None
    for k in TITLE_KEYS:
        v = low_map.get(k)
        if v is not None and str(v).strip():
            title_val = str(v).strip()
            break
    url_val = ""
    for k in URL_KEYS:
        v = low_map.get(k)
        if isinstance(v, str) and v.strip():
            url_val = v.strip()
            break

    return {
        "title": (title_val or default_title)[:240],
        "source": str(low_map.get("source") or default_source)[:80],
        "url": url_val,
        "kind": kind,
        "weight": int(low_map.get("weight") or (6 if kind == "lesson" else 1)),
        "text": text_val,
    }


def _iter_payload_documents(
    raw: bytes,
    fmt: str,
    entry_name: str,
) -> tuple[Iterator[dict[str, Any]], list[int]]:
    """Yield document dicts from a single payload; ``malformed_counter[0]`` tracks bad records."""
    malformed = [0]
    stem = Path(entry_name or "document").stem or "document"
    src = f"import:{stem[:60]}"

    def _gen() -> Iterator[dict[str, Any]]:
        if fmt == "parquet":
            cols, rows = iter_parquet_rows(raw)
            if not cols:
                return
            for idx, r in enumerate(rows):
                d = _extract_record_doc(r, f"{stem} #{idx + 1}", src)
                if d:
                    yield d
            return

        text, _ = _decode_text_bytes(raw)
        if fmt == "jsonl":
            for idx, line in enumerate(text.splitlines()):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    malformed[0] += 1
                    continue
                if isinstance(obj, dict):
                    d = _extract_record_doc(obj, f"{stem} #{idx + 1}", src)
                    if d:
                        yield d
            return

        if fmt == "json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON file: {exc}") from exc
            items = parsed if isinstance(parsed, list) else (
                parsed.get("documents") or parsed.get("items") or parsed.get("rows") or parsed.get("data") or [parsed]
                if isinstance(parsed, dict)
                else []
            )
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    d = _extract_record_doc(item, f"{stem} #{idx + 1}", src)
                    if d:
                        yield d
                elif isinstance(item, str) and item.strip():
                    yield {"title": f"{stem} #{idx + 1}", "source": src, "kind": "page", "weight": 1, "text": item}
            return

        if fmt in ("csv", "tsv"):
            delim = "\t" if fmt == "tsv" else ","
            reader = csv.DictReader(io.StringIO(text), delimiter=delim)
            for idx, row in enumerate(reader):
                clean_row = {str(k or f"col_{i}"): (v or "") for i, (k, v) in enumerate(row.items())}
                d = _extract_record_doc(clean_row, f"{stem} #{idx + 1}", src)
                if d:
                    yield d
            return

        if fmt == "html":
            title_m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
            page_title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else stem
            cleaned = clean_document_text(text, strip_html=True)
            if cleaned:
                yield {"title": page_title or stem, "source": src, "kind": "page", "weight": 1, "text": cleaned}
            return

        if fmt == "xml":
            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                raise ValueError(f"Malformed XML feed: {exc}") from exc
            emitted = 0
            for elem in root.iter():
                tag = elem.tag.split("}")[-1].lower()
                if tag in ("item", "entry", "page", "doc"):
                    children = {c.tag.split("}")[-1].lower(): "".join(c.itertext()).strip() for c in elem}
                    body = children.get("content") or children.get("encoded") or children.get("summary") or children.get("description") or children.get("text") or ""
                    body = clean_document_text(body, strip_html=True)
                    if body:
                        emitted += 1
                        yield {
                            "title": (children.get("title") or f"{stem} #{emitted}")[:240],
                            "source": src,
                            "url": children.get("link") or "",
                            "kind": "page",
                            "weight": 1,
                            "text": body,
                        }
            return

        # Markdown / plain text: keep reasonable document chunks
        cleaned = clean_document_text(text, strip_html=False)
        if cleaned:
            yield {"title": stem[:240], "source": src, "kind": "page", "weight": 1, "text": cleaned}

    return _gen(), malformed


def _iter_container_entries(
    raw: bytes,
    name: str,
    compression_chain: list[str],
    archive_chain: list[str],
    depth: int = 0,
) -> list[tuple[str, bytes, str, str]]:
    """Unwrap compression and archives safely into ``(entry_name, data, fmt, encoding)`` tuples."""
    if depth > MAX_NESTED_DEPTH:
        raise ValueError("Archive nesting depth exceeds maximum safety limit.")

    comp = _detect_compression(raw[:16])
    if comp is not None:
        compression_chain.append(comp)
        inner = _decompress_bytes(raw, comp)
        stem = re.sub(r"\.(gz|bz2|xz|zst)$", "", name, flags=re.I) or "stream"
        return _iter_container_entries(inner, stem, compression_chain, archive_chain, depth + 1)

    if raw[:4] == b"PK\x03\x04":
        archive_chain.append("zip")
        out: list[tuple[str, bytes, str, str]] = []
        total_uncompressed = 0
        src_size = max(1, len(raw))
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ARCHIVE_ENTRIES:
                    raise ValueError(f"Archive contains too many entries ({len(infos)} > {MAX_ARCHIVE_ENTRIES}).")
                for info in infos:
                    if info.is_dir():
                        continue
                    safe_name = _safe_archive_member(info.filename)
                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_DECOMPRESSED_BYTES:
                        raise ValueError("Archive uncompressed payload exceeds maximum allowed size (zip bomb protection).")
                    if info.file_size > 1024 * 1024 and total_uncompressed > src_size * MAX_ARCHIVE_RATIO:
                        raise ValueError("Archive expansion ratio exceeds safety limit (zip bomb protection).")
                    with zf.open(info, "r") as zstream:
                        buf = zstream.read(MAX_DECOMPRESSED_BYTES + 1)
                        if len(buf) > MAX_DECOMPRESSED_BYTES:
                            raise ValueError("Entry exceeds maximum uncompressed size.")
                    out.extend(_iter_container_entries(buf, safe_name, [], archive_chain, depth + 1))
        except (zipfile.BadZipFile, EOFError, OSError, Exception) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Corrupt ZIP archive: {exc}") from exc
        return out

    if _is_tar_bytes(raw):
        archive_chain.append("tar")
        out = []
        total_uncompressed = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
                for idx, member in enumerate(tf):
                    if idx >= MAX_ARCHIVE_ENTRIES:
                        raise ValueError("TAR archive exceeds maximum entry count.")
                    if not member.isfile():
                        continue
                    safe_name = _safe_archive_member(member.name)
                    total_uncompressed += member.size
                    if total_uncompressed > MAX_DECOMPRESSED_BYTES:
                        raise ValueError("TAR uncompressed payload exceeds maximum size.")
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    buf = extracted.read(MAX_DECOMPRESSED_BYTES + 1)
                    out.extend(_iter_container_entries(buf, safe_name, [], archive_chain, depth + 1))
        except tarfile.TarError as exc:
            raise ValueError(f"Corrupt TAR archive: {exc}") from exc
        return out

    fmt, enc = _sniff_inner_format(raw, name)
    return [(name, raw, fmt, enc)]


def inspect_bytes(
    raw: bytes,
    name: str = "file",
    mime: str = "",
    token: str = "",
) -> dict[str, Any]:
    """Inspect a file buffer without writing anything to the corpus."""
    compression: list[str] = []
    archive: list[str] = []
    entries = _iter_container_entries(raw, name, compression, archive)
    if not entries:
        return {
            "token": token,
            "kind": "documents",
            "name": name,
            "mime": mime,
            "bytes": len(raw),
            "format": "empty",
            "compression": compression,
            "archive": archive,
            "entries": 0,
            "encoding": "utf-8",
            "resumable": False,
            "fields": [{"label": "Entries", "value": "0"}],
            "documents": [],
            "warnings": ["Archive contained no readable files."],
            "notes": [],
            "message": "No documents found.",
        }

    # Check if this is a native v3 binary model (.emb)
    if len(entries) == 1 and entries[0][2] == "ember-model":
        from .model import load_model_bytes

        cfg, _, meta = load_model_bytes(entries[0][1])
        has_tok = isinstance(meta.get("tokenizer"), dict)
        return {
            "token": token,
            "kind": "brain",
            "name": name,
            "mime": mime,
            "bytes": len(raw),
            "format": "ember-model",
            "compression": compression,
            "archive": archive,
            "entries": 1,
            "encoding": "binary",
            "resumable": False,
            "legacy": False,
            "fields": [
                {"label": "Format", "value": "Ember v3 binary (.emb)"},
                {"label": "Step", "value": str(meta.get("step", 0))},
                {"label": "Parameters", "value": f"{cfg.n_params:,}"},
                {"label": "Layers / Embd", "value": f"{cfg.n_layer} / {cfg.n_embd}"},
                {"label": "Tokenizer", "value": "Embedded byte-BPE v3" if has_tok else f"vocab {cfg.vocab_size}"},
            ],
            "documents": [],
            "warnings": [],
            "notes": [],
            "message": "Ready to import Ember v3 binary model (.emb).",
        }

    # Check if this is a brain file (v4 current or legacy v1–v3)
    if len(entries) == 1 and entries[0][2] == "brain":
        text, enc = _decode_text_bytes(entries[0][1])
        obj = json.loads(text)
        ver = int(obj.get("version") or 0)
        is_v4 = (obj.get("kind") == BRAIN_KIND and ver == BRAIN_VERSION)
        warnings: list[str] = []
        if not is_v4:
            warnings.append(
                f"Legacy JavaScript brain format (v{ver or 1}) detected. "
                "Weights from the old TensorFlow.js engine are incompatible with the v3 Python/C++ architecture; "
                "embedded corpus documents can still be recovered or you can cast a fresh v3 vessel."
            )
        cfg = obj.get("config") or obj.get("cfg") or {}
        return {
            "token": token,
            "kind": "brain",
            "name": name,
            "mime": mime,
            "bytes": len(raw),
            "format": "brain",
            "compression": compression,
            "archive": archive,
            "entries": 1,
            "encoding": enc,
            "resumable": False,
            "legacy": not is_v4,
            "fields": [
                {"label": "Format", "value": f"Ember v{BRAIN_VERSION}" if is_v4 else f"Legacy JS brain (v{ver or 1})"},
                {"label": "Step", "value": str(obj.get("step") or obj.get("training_state", {}).get("step", 0))},
                {"label": "Layers / Embd", "value": f"{cfg.get('n_layer') or cfg.get('nLayer', '?')} / {cfg.get('n_embd') or cfg.get('nEmbd', '?')}"},
            ],
            "documents": [],
            "warnings": warnings,
            "notes": [],
            "message": "Ready to import Ember brain." if is_v4 else "Incompatible legacy JS brain detected.",
        }

    preview_docs: list[dict[str, Any]] = []
    warnings = []
    total_malformed = 0
    primary_fmt = entries[0][2]
    primary_enc = entries[0][3]
    total_previewed = 0

    for entry_name, payload, fmt, _ in entries:
        it, mal = _iter_payload_documents(payload, fmt, entry_name)
        for doc in it:
            total_previewed += 1
            if len(preview_docs) < 5:
                txt = clean_document_text(doc["text"])
                preview_docs.append(
                    {
                        "title": doc["title"],
                        "source": doc["source"],
                        "chars": len(txt),
                        "excerpt": txt[:220],
                        "entry": entry_name if archive else None,
                    }
                )
            if total_previewed >= 200:
                break
        total_malformed += mal[0]
        if total_previewed >= 200:
            break

    if total_malformed:
        warnings.append(f"Skipped {total_malformed} malformed record(s) during preview.")
    if not preview_docs:
        warnings.append("No text columns or readable documents were found in this file.")

    return {
        "token": token,
        "kind": "documents",
        "name": name,
        "mime": mime,
        "bytes": len(raw),
        "format": primary_fmt,
        "compression": compression,
        "archive": archive,
        "entries": len(entries),
        "encoding": primary_enc,
        "resumable": True,
        "fields": [
            {"label": "Detected format", "value": primary_fmt.upper()},
            {"label": "Sampled records", "value": f"{total_previewed}+" if total_previewed >= 200 else str(total_previewed)},
            {"label": "Encoding", "value": primary_enc},
        ],
        "documents": preview_docs,
        "preview": preview_docs,
        "warnings": warnings,
        "notes": [],
        "message": f"Detected {primary_fmt.upper()} dataset ({len(preview_docs)} previewed).",
    }


@dataclass
class _ImportJob:
    token: str
    path: Path | None
    raw_text: str | None
    name: str
    mime: str
    status: str = "staged"  # staged | running | paused | cancelled | completed | failed
    data_format: str = "text"
    resume_records: int = 0
    accepted: int = 0
    merged: int = 0
    skipped: int = 0
    malformed: int = 0
    records: int = 0
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    pause_flag: bool = False
    cancel_flag: bool = False


class ImportManager:
    """Manages staged files and resumable/cancellable dataset imports."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._lock = threading.Lock()
        self._jobs: dict[str, _ImportJob] = {}
        self._rehydrate_manifest()

    def _rehydrate_manifest(self) -> None:
        raw_rows = self.corpus.get_kv("imports_manifest")
        if not isinstance(raw_rows, list):
            return
        changed = False
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            tok = str(row.get("id") or "").strip()
            if not tok:
                continue
            status = str(row.get("status") or "paused")
            if status == "running":
                status = "paused"
                changed = True
            raw_path = row.get("path")
            p = Path(raw_path) if raw_path else None
            self._jobs[tok] = _ImportJob(
                token=tok,
                path=p,
                raw_text=None,
                name=str(row.get("name") or (p.name if p else "import")),
                mime=str(row.get("mime") or ""),
                status=status,
                data_format=str(row.get("dataFormat") or "text"),
                accepted=int(row.get("documents") or row.get("accepted") or 0),
                merged=int(row.get("merged") or 0),
                skipped=int(row.get("skipped") or 0),
                malformed=int(row.get("malformed") or 0),
                records=int(row.get("records") or 0),
                updated_at=int(row.get("updatedAt") or int(time.time() * 1000)),
            )
        if changed:
            self._persist_manifest()

    def stage_path(self, token: str, path: str | Path, name: str = "", mime: str = "") -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Staged file does not exist: {p}")
        raw = p.read_bytes()
        preview = inspect_bytes(raw, name=name or p.name, mime=mime, token=token)
        with self._lock:
            job = _ImportJob(
                token=token,
                path=p,
                raw_text=None,
                name=name or p.name,
                mime=mime,
                data_format=preview.get("format", "text"),
            )
            self._jobs[token] = job
            self._persist_manifest()
        return preview

    def stage_text(self, token: str, text: str, name: str = "pasted.txt", mime: str = "text/plain") -> dict[str, Any]:
        raw = (text or "").encode("utf-8")
        preview = inspect_bytes(raw, name=name, mime=mime, token=token)
        with self._lock:
            self._jobs[token] = _ImportJob(
                token=token,
                path=None,
                raw_text=text or "",
                name=name,
                mime=mime,
                data_format=preview.get("format", "text"),
            )
            self._persist_manifest()
        return preview

    def commit(
        self,
        token: str,
        path: str | Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            job = self._jobs.get(token)
            if job is None:
                if path is None:
                    raise ValueError(f"Unknown import token: {token}")
                p = Path(path)
                job = _ImportJob(token=token, path=p, raw_text=None, name=p.name, mime="")
                self._jobs[token] = job
            elif path is not None:
                job.path = Path(path)
            if job.path is not None and not job.path.exists():
                job.status = "failed"
                self._persist_manifest()
                raise ValueError(
                    f"Staged import file '{job.name}' is no longer cached on disk. Re-select the file to import."
                )
            job.status = "running"
            job.cancel_flag = False
            job.pause_flag = False
            start_offset = job.records

        raw = job.path.read_bytes() if job.path is not None else (job.raw_text or "").encode("utf-8")
        comp: list[str] = []
        arch: list[str] = []
        entries = _iter_container_entries(raw, job.name, comp, arch)
        decompressed_bytes = sum(len(b) for _, b, _, _ in entries)
        est_records = max(1, decompressed_bytes // 512)

        batch: list[dict[str, Any]] = []
        seen_records = 0
        failed_entries = 0

        for entry_name, payload, fmt, _ in entries:
            try:
                it, mal = _iter_payload_documents(payload, fmt, entry_name)
                for doc in it:
                    if job.cancel_flag or job.pause_flag:
                        break
                    seen_records += 1
                    if seen_records <= start_offset:
                        continue
                    batch.append(doc)
                    if len(batch) >= 100:
                        counts = self.corpus.add_many(batch)
                        batch.clear()
                        job.accepted += counts["accepted"]
                        job.merged += counts["merged"]
                        job.skipped += counts["skipped"]
                        job.records = seen_records
                        job.updated_at = int(time.time() * 1000)
                        if on_progress:
                            pct = min(99, max(1, int(round(100 * seen_records / max(seen_records + 50, est_records)))))
                            on_progress(
                                {
                                    "token": token,
                                    "documents": job.accepted + job.merged,
                                    "processed": job.accepted + job.merged,
                                    "total": max(seen_records, est_records),
                                    "percent": pct,
                                    "stage": "committing",
                                    "records": job.records,
                                    "merged": job.merged,
                                    "skipped": job.skipped,
                                    "failed": failed_entries,
                                    "bytesRead": len(raw),
                                    "sourceBytes": len(raw),
                                    "entries": len(entries),
                                    "elapsedMs": int((time.perf_counter() - t0) * 1000),
                                }
                            )
                job.malformed += mal[0]
            except Exception:
                failed_entries += 1
            if job.cancel_flag or job.pause_flag:
                break

        if batch and not job.cancel_flag:
            counts = self.corpus.add_many(batch)
            job.accepted += counts["accepted"]
            job.merged += counts["merged"]
            job.skipped += counts["skipped"]
            job.records = seen_records

        if job.cancel_flag:
            outcome = "cancelled"
        elif job.pause_flag:
            outcome = "paused"
        else:
            outcome = "completed"
        job.status = outcome
        job.updated_at = int(time.time() * 1000)
        with self._lock:
            self._persist_manifest()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if outcome == "completed":
            final_pct = 100
        else:
            # Paused/cancelled runs keep the honest percentage they reached, so
            # the progress bar never snaps back to 0% after real work was done.
            final_pct = min(99, max(1, int(round(100 * seen_records / max(seen_records + 50, est_records)))))
        if on_progress:
            on_progress(
                {
                    "token": token,
                    "documents": job.accepted + job.merged,
                    "processed": job.accepted + job.merged,
                    "total": max(job.records, est_records),
                    "percent": final_pct,
                    "stage": outcome,
                    "records": job.records,
                    "merged": job.merged,
                    "skipped": job.skipped,
                    "failed": failed_entries,
                    "bytesRead": len(raw),
                    "sourceBytes": len(raw),
                    "entries": len(entries),
                    "elapsedMs": elapsed_ms,
                }
            )
        return {
            "outcome": outcome,
            "documents": job.accepted,
            "accepted": job.accepted,
            "merged": job.merged,
            "skipped": job.skipped,
            "malformed": job.malformed,
            "failedEntries": failed_entries,
            "records": job.records,
            "bytesRead": len(raw),
            "decompressedBytes": decompressed_bytes,
            "elapsedMs": elapsed_ms,
            "resumeRecords": start_offset,
            "format": job.data_format,
            "warnings": [],
        }

    def pause(self, token: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(token)
            if job:
                job.pause_flag = True
                job.status = "paused"
                self._persist_manifest()
        return {"paused": True, "token": token}

    def resume(
        self,
        token: str,
        path: str | Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        return self.commit(token=token, path=path, on_progress=on_progress)

    def cancel(self, token: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(token)
            if job:
                job.cancel_flag = True
                job.status = "cancelled"
                self._persist_manifest()
        return {"cancelled": True, "token": token}

    def discard(self, token: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.pop(token, None)
            if job:
                job.cancel_flag = True
            self._persist_manifest()
        return {"discarded": True, "token": token}

    def list_imports(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": j.token,
                    "name": j.name,
                    "status": j.status,
                    "dataFormat": j.data_format,
                    "documents": j.accepted,
                    "updatedAt": j.updated_at,
                    "resumable": j.path is not None and j.path.exists(),
                }
                for j in self._jobs.values()
                if j.status in ("paused", "partial", "failed", "running")
            ]

    def _persist_manifest(self) -> None:
        rows = [
            {
                "id": j.token,
                "name": j.name,
                "mime": j.mime,
                "path": str(j.path) if j.path is not None else None,
                "status": j.status,
                "dataFormat": j.data_format,
                "documents": j.accepted,
                "accepted": j.accepted,
                "merged": j.merged,
                "skipped": j.skipped,
                "malformed": j.malformed,
                "records": j.records,
                "updatedAt": j.updated_at,
                "resumable": j.path is not None and j.path.exists(),
            }
            for j in self._jobs.values()
        ]
        self.corpus.set_kv("imports_manifest", rows)
