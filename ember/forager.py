"""Background public-web forager implemented in pure Python stdlib."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

from .corpus import Corpus, clean_document_text

USER_AGENT = "EmberFoundry/0.4 (+https://github.com/kilkproduct/Ember)"


def _http_json(url: str, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class Forager:
    """Walks public encyclopedic and archive APIs on a background thread."""

    def __init__(
        self,
        corpus: Corpus,
        emit_event: Callable[[str, dict[str, Any]], None],
        auto_fire_cb: Callable[[], None] | None = None,
    ) -> None:
        self.corpus = corpus
        self._emit = emit_event
        self._auto_fire_cb = auto_fire_cb
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.documents_accepted = 0
        self.last_document = ""
        self.current_seeds = ""
        self.started_at = 0
        self.error: str | None = None
        self.saved_state: dict[str, Any] | None = corpus.get_kv("forager:autosave")
        self.logs: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "documentsAccepted": self.documents_accepted,
            "lastDocument": self.last_document,
            "currentSeeds": self.current_seeds,
            "startedAt": self.started_at,
            "error": self.error,
            "autosave": self.saved_state,
        }

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.running:
                return {"running": True}
            seeds_raw = str(options.get("seeds") or "").strip()
            subs_raw = str(options.get("subreddits") or "").strip()
            if not seeds_raw and not subs_raw:
                seeds_raw = "astronomy, botany, oceanography, history of science"
            self._stop.clear()
            self.running = True
            self.documents_accepted = 0
            self.current_seeds = seeds_raw
            self.started_at = int(time.time() * 1000)
            self.error = None
            opts = dict(options)
            opts["seeds"] = seeds_raw
            t = threading.Thread(target=self._worker, args=(opts,), name="ember-forager", daemon=True)
            self._thread = t
            t.start()
        return {"started": True, "seeds": seeds_raw}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        return {"stopping": True}

    def _log(self, line: str) -> None:
        entry = {"at": int(time.time() * 1000), "line": line}
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        self._emit("forage.log", entry)

    def _accept(self, title: str, text: str, source: str, url: str = "", meta: dict | None = None) -> bool:
        cleaned = clean_document_text(text, strip_html=True)
        if len(cleaned) < 120:
            return False
        doc = self.corpus.add(
            text=cleaned,
            title=title[:240],
            source=source,
            url=url,
            weight=1,
            kind="page",
            meta=meta,
        )
        if not doc.get("merged"):
            self.documents_accepted += 1
            self.last_document = doc.get("title", title)
            self._log(f"+ [{source}] {self.last_document} ({len(cleaned)} chars)")
            self._emit(
                "forage.doc",
                {
                    "title": self.last_document,
                    "source": source,
                    "url": url,
                    "chars": len(cleaned),
                    "documentsAccepted": self.documents_accepted,
                },
            )
            self._emit("corpus.changed", self.corpus.summary())
            return True
        return False

    def _worker(self, opts: dict[str, Any]) -> None:
        budget = max(1, min(500, int(opts.get("budget") or 24)))
        seeds = [s.strip() for s in re.split(r"[,\n;]+", str(opts.get("seeds") or "")) if s.strip()]
        subs = [s.strip().lstrip("r/") for s in re.split(r"[,\s]+", str(opts.get("subreddits") or "")) if s.strip()]
        seen_topics: set[str] = set()
        queue = list(seeds)
        autopilot = bool(opts.get("autopilot", True))

        try:
            self._log(f"Starting forage (budget={budget}, seeds={', '.join(queue[:5])})")
            while queue and self.documents_accepted < budget and not self._stop.is_set():
                topic = queue.pop(0)
                if topic.lower() in seen_topics:
                    continue
                seen_topics.add(topic.lower())
                self.current_seeds = topic

                if opts.get("wikiOn", True) and not self._stop.is_set():
                    self._fetch_wikipedia(topic, queue if (autopilot and opts.get("wander", True)) else None)

                if (opts.get("hn", True)) and self.documents_accepted < budget and not self._stop.is_set():
                    self._fetch_hn(topic)

                if (opts.get("gutenberg", True)) and self.documents_accepted < budget and not self._stop.is_set():
                    self._fetch_gutendex(topic)

                if (opts.get("openlibrary", True)) and self.documents_accepted < budget and not self._stop.is_set():
                    self._fetch_openlibrary(topic)

                for sub in subs[:3]:
                    if self.documents_accepted >= budget or self._stop.is_set():
                        break
                    self._fetch_reddit_arctic(sub, topic)

                state_blob = {
                    "running": not self._stop.is_set(),
                    "documentsAccepted": self.documents_accepted,
                    "lastDocument": self.last_document,
                    "currentSeeds": ", ".join(queue[:6]) or topic,
                    "startedAt": self.started_at,
                    "updatedAt": int(time.time() * 1000),
                    "options": opts,
                }
                self.saved_state = state_blob
                self.corpus.set_kv("forager:autosave", state_blob)
        except Exception as exc:
            self.error = str(exc)
            self._log(f"! forage error: {exc}")
        finally:
            self.running = False
            if self.saved_state:
                self.saved_state["running"] = False
                self.corpus.set_kv("forager:autosave", self.saved_state)
            self._log(f"Foraging finished ({self.documents_accepted} new documents).")
            payload = {
                "documentsAccepted": self.documents_accepted,
                "accepted": self.documents_accepted,
                "stopped": self._stop.is_set(),
                "error": self.error,
                "corpus": self.corpus.summary(),
            }
            self._emit("forage.finished", payload)
            self._emit("forage.done", payload)
            if opts.get("autoFire") and self.documents_accepted > 0 and not self._stop.is_set() and self._auto_fire_cb:
                try:
                    self._auto_fire_cb()
                except Exception:
                    pass

    def _fetch_wikipedia(self, topic: str, wander_queue: list[str] | None) -> None:
        langs = ["simple", "en"]
        for lang in langs:
            if self._stop.is_set():
                return
            try:
                q = urllib.parse.quote(topic)
                search_url = (
                    f"https://{lang}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={q}&srlimit=3&format=json"
                )
                data = _http_json(search_url)
                hits = data.get("query", {}).get("search", [])
                for hit in hits:
                    if self._stop.is_set():
                        return
                    title = hit.get("title", "")
                    if not title:
                        continue
                    ext_url = (
                        f"https://{lang}.wikipedia.org/w/api.php?"
                        f"action=query&prop=extracts|links&explaintext=1&pllimit=8&titles={urllib.parse.quote(title)}&format=json"
                    )
                    pages = _http_json(ext_url).get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = page.get("extract") or ""
                        if extract:
                            self._accept(
                                title,
                                extract,
                                f"wiki:{lang}",
                                url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                            )
                        if wander_queue is not None and len(wander_queue) < 40:
                            for lnk in page.get("links", [])[:3]:
                                lt = lnk.get("title", "")
                                if lt and ":" not in lt:
                                    wander_queue.append(lt)
            except Exception as exc:
                self._log(f"  wiki ({lang}) skipped '{topic}': {exc}")

    def _fetch_hn(self, topic: str) -> None:
        try:
            url = f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(topic)}&tags=comment&hitsPerPage=4"
            data = _http_json(url)
            for hit in data.get("hits", []):
                body = hit.get("comment_text") or hit.get("story_text") or ""
                if body:
                    self._accept(
                        hit.get("story_title") or f"HN: {topic}",
                        body,
                        "hackernews",
                        url=f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    )
        except Exception as exc:
            self._log(f"  hn skipped '{topic}': {exc}")

    def _fetch_gutendex(self, topic: str) -> None:
        try:
            url = f"https://gutendex.com/books?search={urllib.parse.quote(topic)}"
            data = _http_json(url)
            for book in data.get("results", [])[:2]:
                summaries = book.get("summaries") or []
                subjects = ", ".join(book.get("subjects") or [])
                text = "\n\n".join(summaries)
                if len(text) >= 120:
                    self._accept(
                        book.get("title") or topic,
                        f"{book.get('title', '')}\nSubjects: {subjects}\n\n{text}",
                        "gutenberg",
                        url=f"https://www.gutenberg.org/ebooks/{book.get('id', '')}",
                    )
        except Exception as exc:
            self._log(f"  gutenberg skipped '{topic}': {exc}")

    def _fetch_openlibrary(self, topic: str) -> None:
        try:
            url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(topic)}&limit=3"
            data = _http_json(url)
            for doc in data.get("docs", [])[:3]:
                first_sentence = doc.get("first_sentence")
                if isinstance(first_sentence, list):
                    first_sentence = " ".join(first_sentence)
                subj = ", ".join((doc.get("subject") or [])[:8])
                txt = f"{doc.get('title', '')}. {first_sentence or ''}\nSubjects: {subj}".strip()
                if len(txt) >= 120:
                    key = doc.get("key") or ""
                    self._accept(doc.get("title") or topic, txt, "openlibrary", url=f"https://openlibrary.org{key}")
        except Exception as exc:
            self._log(f"  openlibrary skipped '{topic}': {exc}")

    def _fetch_reddit_arctic(self, subreddit: str, topic: str) -> None:
        try:
            url = (
                "https://arctic-shift.photon-reddit.com/api/posts/search?"
                f"subreddit={urllib.parse.quote(subreddit)}&title={urllib.parse.quote(topic)}&limit=3"
            )
            data = _http_json(url)
            for post in data.get("data", [])[:3]:
                body = post.get("selftext") or ""
                title = post.get("title") or ""
                if len(body) >= 120:
                    self._accept(
                        title or f"r/{subreddit}",
                        f"[r/{subreddit}] {title}\n\n{body}",
                        f"r/{subreddit}",
                        url=f"https://www.reddit.com/r/{subreddit}/comments/{post.get('id', '')}/",
                        meta={"subreddit": subreddit},
                    )
        except Exception as exc:
            self._log(f"  reddit r/{subreddit} skipped: {exc}")
