"""Cloud teacher desk and GitHub brain sync in pure Python stdlib."""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .corpus import Corpus

WINDOWS = [
    {"id": "daily", "label": "Daily", "ms": 24 * 3600 * 1000},
    {"id": "5h", "label": "5 hours", "ms": 5 * 3600 * 1000},
    {"id": "12h", "label": "12 hours", "ms": 12 * 3600 * 1000},
    {"id": "weekly", "label": "Weekly", "ms": 7 * 24 * 3600 * 1000},
    {"id": "monthly", "label": "Monthly", "ms": 30 * 24 * 3600 * 1000},
    {"id": "session", "label": "This session", "ms": 0},
]

TEACHER_DEFS = [
    {
        "id": "pollinations",
        "name": "Pollinations",
        "compute": "Cloudflare edge (anonymous text)",
        "free": True,
        "needs": [],
        "defaultMax": 80,
        "model": "openai",
        "hint": "No key required. Spaced automatically to respect the public rate limit.",
    },
    {
        "id": "groq",
        "name": "Groq",
        "compute": "Groq LPU (free tier)",
        "free": True,
        "needs": ["token"],
        "defaultMax": 60,
        "model": "llama-3.1-8b-instant",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "hint": "API key from console.groq.com.",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "compute": "OpenRouter free models",
        "free": True,
        "needs": ["token"],
        "defaultMax": 50,
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "hint": "API key from openrouter.ai/keys.",
    },
    {
        "id": "cloudflare",
        "name": "Cloudflare Workers AI",
        "compute": "Cloudflare GPU (Workers AI)",
        "free": True,
        "needs": ["accountId", "token"],
        "defaultMax": 100,
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "hint": "Account ID + API token from dash.cloudflare.com.",
    },
]


class TeacherDesk:
    """Manages teacher configs, quota counters, single drafts, and auto-teach runs."""

    def __init__(self, corpus: Corpus, emit_event: Callable[[str, dict[str, Any]], None]) -> None:
        self.corpus = corpus
        self._emit = emit_event
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.running = False
        saved = corpus.get_kv("teachers:state")
        self.state: dict[str, dict[str, Any]] = saved if isinstance(saved, dict) else {}
        self.logs: list[dict[str, Any]] = []

    def _log(self, line: str) -> None:
        entry = {"at": int(time.time() * 1000), "line": line}
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        self._emit("teachers.log", entry)

    def _window_ms(self, win_id: str) -> int:
        for w in WINDOWS:
            if w["id"] == win_id:
                return int(w["ms"])
        return 24 * 3600 * 1000

    def _entry_state(self, tdef: dict[str, Any]) -> dict[str, Any]:
        s = dict(self.state.get(tdef["id"]) or {})
        now_ms = int(time.time() * 1000)
        win = str(s.get("window") or "daily")
        reset_at = int(s.get("resetAt") or 0)
        if reset_at and now_ms >= reset_at:
            s["used"] = 0
            s["exhausted"] = False
            s["resetAt"] = now_ms + self._window_ms(win) if self._window_ms(win) > 0 else 0
        s.setdefault("on", False)
        s.setdefault("window", win)
        s.setdefault("max", int(tdef.get("defaultMax", 40)))
        s.setdefault("used", 0)
        s.setdefault("exhausted", False)
        s.setdefault("lastError", "")
        s.setdefault("accountId", "")
        s.setdefault("token", "")
        s.setdefault("resetAt", reset_at)
        self.state[tdef["id"]] = s
        return s

    def descriptors(self) -> list[dict[str, Any]]:
        out = []
        for t in TEACHER_DEFS:
            s = self._entry_state(t)
            rem = max(0, int(s["max"]) - int(s["used"]))
            out.append(
                {
                    "id": t["id"],
                    "name": t["name"],
                    "compute": t["compute"],
                    "free": bool(t["free"]),
                    "needs": list(t["needs"]),
                    "hint": t["hint"],
                    "defaultMax": int(t["defaultMax"]),
                    "model": t["model"],
                    "state": s,
                    "remaining": rem,
                    "resetAt": int(s["resetAt"]),
                }
            )
        return out

    def enabled_ids(self) -> list[str]:
        return [d["id"] for d in self.descriptors() if d["state"]["on"]]

    def _emit_state(self) -> None:
        self._emit("teachers.state", {"teachers": self.descriptors(), "running": self.running})

    def update(self, teacher_id: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
        tdef = next((t for t in TEACHER_DEFS if t["id"] == teacher_id), None)
        if tdef is None:
            raise ValueError(f"Unknown teacher '{teacher_id}'.")
        with self._lock:
            s = self._entry_state(tdef)
            for k in ("on", "window", "max", "accountId", "token"):
                if k in fields:
                    s[k] = fields[k]
            if fields.get("on") is True and int(s["used"]) < int(s["max"]):
                s["exhausted"] = False
            if "window" in fields:
                ms = self._window_ms(str(fields["window"]))
                s["resetAt"] = int(time.time() * 1000) + ms if ms > 0 else 0
            self.state[teacher_id] = s
            self.corpus.set_kv("teachers:state", self.state)
        return self.descriptors()

    def set_default_window(self, window: str) -> list[dict[str, Any]]:
        with self._lock:
            for t in TEACHER_DEFS:
                s = self._entry_state(t)
                if not s.get("on"):
                    s["window"] = window
            self.corpus.set_kv("teachers:state", self.state)
        return self.descriptors()

    def _call_provider(self, tdef: dict[str, Any], s: dict[str, Any], topic: str) -> str:
        prompt = (
            f"Write a clear, concise, factual educational passage (2 to 4 paragraphs) about '{topic}'. "
            "Plain prose, no disclaimers."
        )
        tid = tdef["id"]
        if tid == "pollinations":
            url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}"
            req = urllib.request.Request(url, headers={"User-Agent": "EmberFoundry/0.4"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8", errors="replace").strip()

        if tid == "cloudflare":
            acct = str(s.get("accountId") or "").strip()
            tok = str(s.get("token") or "").strip()
            if not acct or not tok:
                raise ValueError("Set Cloudflare Account ID and API token first.")
            url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{tdef['model']}"
            payload = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                return str(data.get("result", {}).get("response") or "").strip()

        tok = str(s.get("token") or "").strip()
        if not tok:
            raise ValueError(f"Set API key for {tdef['name']} first.")
        payload = json.dumps(
            {
                "model": tdef["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            tdef["url"],
            data=payload,
            headers={
                "Authorization": f"Bearer {tok}",
                "Content-Type": "application/json",
                "User-Agent": "EmberFoundry/0.4",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return str(data["choices"][0]["message"]["content"]).strip()

    def ask(self, teacher_id: str, topic: str) -> dict[str, Any]:
        tdef = next((t for t in TEACHER_DEFS if t["id"] == teacher_id), None)
        if tdef is None:
            raise ValueError(f"Unknown teacher '{teacher_id}'.")
        topic_clean = (topic or "").strip() or "foundations of natural science"
        s = self._entry_state(tdef)
        if int(s["used"]) >= int(s["max"]):
            s["exhausted"] = True
            s["on"] = False
            self.corpus.set_kv("teachers:state", self.state)
            why = f"Quota cap reached ({s['used']}/{s['max']})."
            self._emit("teachers.exhausted", {"id": teacher_id, "name": tdef["name"], "why": why})
            self._emit_state()
            raise ValueError(f"{tdef['name']} quota reached ({s['used']}/{s['max']}).")
        try:
            text = self._call_provider(tdef, s, topic_clean)
            if not text:
                raise ValueError("Teacher returned an empty response.")
            s["used"] = int(s["used"]) + 1
            s["lastError"] = ""
            if int(s["used"]) >= int(s["max"]):
                s["exhausted"] = True
                s["on"] = False
                self._emit(
                    "teachers.exhausted",
                    {"id": teacher_id, "name": tdef["name"], "why": f"Reached window cap ({s['max']})."},
                )
            self.corpus.set_kv("teachers:state", self.state)
            return {
                "id": teacher_id,
                "teacher": tdef["name"],
                "topic": topic_clean,
                "title": f"{topic_clean} ({tdef['name']})",
                "text": text,
                "chars": len(text),
            }
        except urllib.error.HTTPError as exc:
            s["lastError"] = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (401, 402, 403, 429):
                s["exhausted"] = True
                s["on"] = False
                self.corpus.set_kv("teachers:state", self.state)
                self._emit(
                    "teachers.exhausted",
                    {"id": teacher_id, "name": tdef["name"], "why": f"Provider returned HTTP {exc.code}."},
                )
                self._emit_state()
            else:
                self.corpus.set_kv("teachers:state", self.state)
            raise
        except Exception as exc:
            s["lastError"] = str(exc)
            self.corpus.set_kv("teachers:state", self.state)
            raise

    def accept_draft(self, teacher_id: str, topic: str, text: str) -> dict[str, Any]:
        doc = self.corpus.add(
            text=text,
            title=(topic or "Teacher lesson")[:240],
            source=f"teacher:{teacher_id or 'cloud'}",
            kind="lesson",
            weight=4,
        )
        self._emit("corpus.changed", self.corpus.summary())
        return doc

    def run_auto(self, topics_str: str) -> dict[str, Any]:
        enabled = [d for d in self.descriptors() if d["state"]["on"]]
        if not enabled:
            raise ValueError("Enable at least one teacher before starting auto-teach.")
        if self.running:
            return {"running": True}
        topics = [t.strip() for t in (topics_str or "").replace("\n", ",").split(",") if t.strip()]
        if not topics:
            topics = ["astronomy", "world history", "biology", "physics"]
        self._stop.clear()
        self.running = True
        self._emit_state()

        def _worker() -> None:
            accepted = 0
            try:
                for idx, topic in enumerate(topics):
                    if self._stop.is_set():
                        break
                    tdesc = enabled[idx % len(enabled)]
                    try:
                        draft = self.ask(tdesc["id"], topic)
                        self.accept_draft(tdesc["id"], topic, draft["text"])
                        accepted += 1
                        self._log(f"+ [{tdesc['name']}] {topic} ({draft['chars']} chars)")
                    except Exception as exc:
                        self._log(f"! [{tdesc['name']}] {topic}: {exc}")
            finally:
                self.running = False
                self._emit_state()
                self._emit("teachers.done", {"accepted": accepted})

        threading.Thread(target=_worker, name="ember-teachers", daemon=True).start()
        return {"started": True}

    def stop_auto(self) -> dict[str, Any]:
        self._stop.set()
        return {"stopped": True}


# --------------------------------------------------------------------------
# GitHub Contents API sync for portable Ember v4 brains
# --------------------------------------------------------------------------

def github_head(owner: str, repo: str, path: str, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "EmberFoundry/0.4",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return {"sha": data.get("sha"), "size": data.get("size", 0)}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"sha": None, "size": 0}
        raise ValueError(f"GitHub API HTTP {exc.code}: {exc.reason}") from exc


def github_pull_brain(owner: str, repo: str, path: str, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "EmberFoundry/0.4",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            if "content" in data and data["content"]:
                raw = base64.b64decode(data["content"]).decode("utf-8")
            elif data.get("download_url"):
                dl_req = urllib.request.Request(
                    data["download_url"],
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "EmberFoundry/0.4"},
                )
                with urllib.request.urlopen(dl_req, timeout=60) as dl_resp:
                    raw = dl_resp.read().decode("utf-8")
            else:
                return {"brain": None, "sha": None}
            return {"brain": json.loads(raw), "sha": data.get("sha")}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"brain": None, "sha": None}
        raise ValueError(f"GitHub pull failed (HTTP {exc.code}): {exc.reason}") from exc


def github_push_brain(owner: str, repo: str, path: str, token: str, brain: dict[str, Any], sha: str | None) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    encoded = base64.b64encode(json.dumps(brain, ensure_ascii=True).encode("utf-8")).decode("ascii")
    body: dict[str, Any] = {
        "message": f"Update Ember brain ({brain.get('name', 'checkpoint')})",
        "content": encoded,
    }
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "EmberFoundry/0.4",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return str(data.get("content", {}).get("sha") or "")
    except urllib.error.HTTPError as exc:
        raise ValueError(f"GitHub push failed (HTTP {exc.code}): {exc.reason}") from exc
