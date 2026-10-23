"""Background public-web forager implemented in pure Python stdlib."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import json
import random
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from .progress import WorkProgress
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

from .corpus import Corpus, clean_document_text

USER_AGENT = "EmberFoundry/0.4 (+https://github.com/kilkproduct/Ember)"


def _http_json(url: str, timeout: float = 8.0, on_bytes=None) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data=bytearray()
        while True:
            chunk=resp.read(64*1024)
            if not chunk:break
            data.extend(chunk)
            if on_bytes:on_bytes(len(chunk))
            if len(data)>8*1024*1024:raise ValueError("API response exceeds 8 MiB limit")
        return json.loads(data.decode("utf-8",errors="strict"))


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
        self._lock = threading.RLock()
        self._host_slots=defaultdict(lambda:threading.BoundedSemaphore(2))
        self._host_next=defaultdict(float)
        self._wiki_languages=["simple","en"]
        self._cache_dir=corpus.db_path.parent/"forage_http_cache"
        self._cache_dir.mkdir(exist_ok=True)
        self.metrics={k:0 for k in ("activeRequests","cached","fetched","failed","duplicates","rejected","bytesDownloaded","completedSources")}
        self.progress=WorkProgress()
        self._budget=500
        self._last_summary=0.0
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
        with self._lock:
            return {
            **self.metrics,
            "progress": self.progress.update(self.metrics["completedSources"],state="running" if self.running else "stopped"),
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
            if options.get("resume") and not (self.saved_state or {}).get("pendingTopics"):
                raise ValueError("No pending foraging topics to resume")
            self._wiki_languages=["simple","en"] if options.get("wikiFamily",True) else ["en"]
            seeds_raw = str(options.get("seeds") or "").strip()
            extra=str(options.get("wiki") or "").strip()
            if extra:seeds_raw=", ".join(filter(None,[seeds_raw,extra]))
            subs_raw = str(options.get("subreddits") or "").strip()
            if not seeds_raw and not subs_raw:
                seeds_raw = "astronomy, botany, oceanography, history of science"
            self.metrics={k:0 for k in self.metrics}
            self.progress=WorkProgress()
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

    def _cached(self,url):
        path=self._cache_dir/(hashlib.sha256(url.encode()).hexdigest()+".json")
        try:
            if path.stat().st_size>8*1024*1024 or time.time()-path.stat().st_mtime>3600:return None
            return json.loads(path.read_text(encoding="utf-8"))["payload"]
        except (OSError,ValueError,KeyError):return None

    def _cache(self,url,payload):
        # Disposable public-response cache. Bound both entries and disk bytes;
        # files are atomically replaced, never touch models/corpus/checkpoints.
        raw=json.dumps({"payload":payload},ensure_ascii=False).encode("utf-8")
        if len(raw)>8*1024*1024:return
        path=self._cache_dir/(hashlib.sha256(url.encode()).hexdigest()+".json")
        try:
            with self._lock:
                temporary=path.with_suffix(".tmp")
                temporary.write_bytes(raw);os.replace(temporary,path)
                entries=sorted(self._cache_dir.glob("*.json"),key=lambda p:p.stat().st_mtime)
                size=sum(p.stat().st_size for p in entries)
                while entries and (len(entries)>64 or size>16*1024*1024):
                    oldest=entries.pop(0);size-=oldest.stat().st_size;oldest.unlink()
        except OSError:pass # cache failures cannot turn successful retrieval into failure

    def _request(self,url):
        if self._stop.is_set():raise InterruptedError("Forage cancelled")
        cached=self._cached(url)
        if cached is not None:
            with self._lock:self.metrics["cached"]+=1
            return cached
        host=urllib.parse.urlsplit(url).netloc
        slot=self._host_slots[host]
        while not self._stop.is_set():
            if slot.acquire(timeout=.1):break
        else:raise InterruptedError("Forage cancelled")
        def received(n):
            with self._lock:self.metrics["bytesDownloaded"]+=n
            if self._stop.is_set():raise InterruptedError("Forage cancelled")
        try:
            with self._lock:self.metrics["activeRequests"]+=1
            for attempt in range(3):
                if self._stop.is_set():raise InterruptedError("Forage cancelled")
                try:
                    with self._lock:
                        now=time.monotonic();due=max(now,self._host_next[host]);self._host_next[host]=due+.5
                    if self._stop.wait(max(0,due-now)):raise InterruptedError("Forage cancelled")
                    result=_http_json(url,on_bytes=received)
                    self._cache(url,result)
                    with self._lock:self.metrics["fetched"]+=1
                    return result
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429,500,502,503,504) or attempt==2:raise
                    retry=exc.headers.get("Retry-After","") if exc.headers else ""
                    delay=min(30,float(retry)) if retry.isdigit() else 2**attempt+random.random()*.2
                    if self._stop.wait(delay):raise InterruptedError("Forage cancelled")
                except (TimeoutError,urllib.error.URLError):
                    if attempt==2:raise
                    if self._stop.wait(2**attempt):raise InterruptedError("Forage cancelled")
        except InterruptedError:
            raise
        except Exception:
            with self._lock:self.metrics["failed"]+=1
            raise
        finally:
            with self._lock:self.metrics["activeRequests"]-=1
            slot.release()

    def _accept(self,*args,**kwargs):
        with self._lock:
            if self._stop.is_set() or self.documents_accepted>=self._budget:return False
            return self._accept_locked(*args,**kwargs)

    def _accept_locked(self, title: str, text: str, source: str, url: str = "", meta: dict | None = None) -> bool:
        cleaned = clean_document_text(text, strip_html=True)
        if len(cleaned) < 120:
            self.metrics["rejected"]+=1
            return False
        doc = self.corpus.add(
            text=cleaned,
            title=title[:240],
            source=source,
            url=url,
            weight=1,
            kind="page",
            meta={**(meta or {}),"retrievedAt":int(time.time()),"extraction":"public-api"},
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
            now=time.monotonic()
            if now-self._last_summary>=.5:
                self._last_summary=now
                self._emit("corpus.changed", self.corpus.summary())
            return True
        self.metrics["duplicates"]+=1
        return False

    def _worker(self, opts: dict[str, Any]) -> None:
        budget = max(1, min(500, int(opts.get("budget") or 24)))
        seeds = [s.strip() for s in re.split(r"[,\n;]+", str(opts.get("seeds") or "")) if s.strip()]
        subs = [s.strip().lstrip("r/") for s in re.split(r"[,\s]+", str(opts.get("subreddits") or "")) if s.strip()]
        self._budget=budget
        resume=self.saved_state if opts.get("resume") else None
        seen_topics=set(resume.get("seenTopics",[])) if resume else set()
        queue=list(resume.get("pendingTopics",seeds)) if resume else list(seeds)
        autopilot = bool(opts.get("autopilot", True))

        try:
            self._log(f"Starting forage (budget={budget}, seeds={', '.join(queue[:5])})")
            while queue and self.documents_accepted < budget and not self._stop.is_set():
                topic = queue.pop(0)
                if topic.lower() in seen_topics:
                    continue
                self.current_seeds = topic
                # Persist in-flight work before requests. Cancellation/crash must
                # replay this topic; corpus dedup prevents repeated content.
                self.saved_state={"running":True,"pendingTopics":[topic]+list(queue),
                    "seenTopics":sorted(seen_topics),"options":opts}
                self.corpus.set_kv("forager:autosave",self.saved_state)

                jobs=[]
                if opts.get("wikiOn",True):jobs.append((self._fetch_wikipedia,(topic,queue if autopilot and opts.get("wander",True) else None)))
                for key,fn in (("hn",self._fetch_hn),("gutenberg",self._fetch_gutendex),("openlibrary",self._fetch_openlibrary),("stack",self._fetch_stack)):
                    if opts.get(key, key!="stack"):jobs.append((fn,(topic,)))
                for sub in subs[:3]:
                    if opts.get("redditArctic",True):jobs.append((self._fetch_reddit_arctic,(sub,topic)))
                    if opts.get("redditPullpush",False):jobs.append((self._fetch_reddit_pullpush,(sub,topic)))
                # Sources use different hosts. Each host has a separate 2-request
                # bound; writes/budget checks are serialized and updates throttled.
                with ThreadPoolExecutor(max_workers=4,thread_name_prefix="ember-fetch") as pool:
                    futures=[pool.submit(fn,*args) for fn,args in jobs if not self._stop.is_set()]
                    for future in futures:
                        future.result()
                        with self._lock:self.metrics["completedSources"]+=1
                        self._emit("forage.progress",self.status())
                if self._stop.is_set() or self.documents_accepted>=budget:
                    pending=[topic]+list(queue)
                else:
                    seen_topics.add(topic.lower());pending=list(queue)
                state_blob = {
                    "running": not self._stop.is_set(),
                    "documentsAccepted": self.documents_accepted,
                    "lastDocument": self.last_document,
                    "currentSeeds": ", ".join(queue[:6]) or topic,
                    "startedAt": self.started_at,
                    "updatedAt": int(time.time() * 1000),
                    "options": opts,
                    "pendingTopics":pending,
                    "seenTopics":sorted(seen_topics),
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
            self._emit("corpus.changed",self.corpus.summary())
            self._emit("forage.finished", payload)
            self._emit("forage.done", payload)
            if opts.get("autoFire") and self.documents_accepted > 0 and not self._stop.is_set() and self._auto_fire_cb:
                try:
                    self._auto_fire_cb()
                except Exception:
                    pass

    def _fetch_wikipedia(self, topic: str, wander_queue: list[str] | None) -> None:
        langs = self._wiki_languages
        for lang in langs:
            if self._stop.is_set():
                return
            try:
                q = urllib.parse.quote(topic)
                search_url = (
                    f"https://{lang}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={q}&srlimit=3&format=json"
                )
                data = self._request(search_url)
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
                    pages = self._request(ext_url).get("query", {}).get("pages", {})
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
            data = self._request(url)
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
            data = self._request(url)
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
            data = self._request(url)
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
            data = self._request(url)
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

    def _fetch_stack(self,topic):
        try:
            url="https://api.stackexchange.com/2.3/search/advanced?site=stackoverflow&filter=withbody&pagesize=3&q="+urllib.parse.quote(topic)
            data=self._request(url)
            if data.get("backoff"):
                self._stop.wait(min(30,float(data["backoff"])))
            for item in data.get("items",[]):
                self._accept(item.get("title",topic),item.get("body", ""),"stackoverflow",item.get("link",""))
        except Exception as exc:self._log(f"Stack Exchange unavailable: {exc}")

    def _fetch_reddit_pullpush(self,subreddit,topic):
        try:
            url="https://api.pullpush.io/reddit/search/submission/?size=3&subreddit="+urllib.parse.quote(subreddit)+"&q="+urllib.parse.quote(topic)
            for item in self._request(url).get("data",[]):
                self._accept(item.get("title",topic),item.get("selftext", ""),"reddit-pullpush", "https://www.reddit.com"+item.get("permalink",""))
        except Exception as exc:self._log(f"PullPush unavailable: {exc}")
