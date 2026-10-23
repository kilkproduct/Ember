"""Static contract tests for the browser presentation client (`web/`).

These tests exist because the web UI silently died twice in this repository's
history:

1. ``app.js`` imported a name that ``notify.js`` never exported.  In an ES
   module that is a link-time ``SyntaxError``, so *nothing* in the page ran —
   no tabs, no buttons, no status.
2. ``app.js`` addressed elements by ids that do not exist in ``index.html``
   (``#btn-cast`` instead of ``#btn-init``, ``#gen-max`` instead of
   ``#gen-tokens``, …).  Every one of those controls was inert.

Both classes of failure are invisible to a Python test suite unless the client
is checked against the files it actually talks to, which is what this module
does.  It also verifies that every RPC method the client calls is dispatched by
``ember.backend`` and that the service worker caches only files that exist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ember.backend import EmberBackend

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

MODULES = ["app.js", "notify.js", "help.js", "lifecycle.js", "sw.js"]


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _exported_names(source: str) -> set[str]:
    names: set[str] = set()
    names.update(re.findall(r"export\s+(?:async\s+)?function\s+([A-Za-z0-9_$]+)", source))
    names.update(re.findall(r"export\s+(?:const|let|var)\s+([A-Za-z0-9_$]+)", source))
    names.update(re.findall(r"export\s+class\s+([A-Za-z0-9_$]+)", source))
    for group in re.findall(r"export\s*\{([^}]*)\}", source):
        for part in group.split(","):
            part = part.strip()
            if not part:
                continue
            names.add(part.split(" as ")[-1].strip())
    return names


def _imported_pairs(source: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for names_blob, spec in re.findall(r"import\s*\{([^}]*)\}\s*from\s*[\"']([^\"']+)[\"']", source):
        names = [n.strip().split(" as ")[0].strip() for n in names_blob.split(",") if n.strip()]
        out.append((spec, names))
    return out


def test_every_web_module_exists_and_is_referenced_by_index_or_app() -> None:
    index = _read("index.html")
    assert 'src="app.js"' in index, "index.html must load app.js as a module"
    for name in MODULES:
        assert (WEB_DIR / name).exists(), f"web/{name} is missing"


def test_no_module_imports_a_name_that_is_not_exported() -> None:
    """Regression: `import { installNotifications } from "./notify.js"` killed the whole UI."""
    problems: list[str] = []
    for name in MODULES:
        source = _read(name)
        for spec, names in _imported_pairs(source):
            if not spec.startswith("./"):
                continue
            target = (WEB_DIR / spec[2:]).resolve()
            if not target.exists():
                problems.append(f"{name}: imports missing module {spec}")
                continue
            exported = _exported_names(target.read_text(encoding="utf-8"))
            for imported in names:
                if imported not in exported:
                    problems.append(f"{name}: '{imported}' is not exported by {spec}")
    assert not problems, "Broken ES module imports (link-time SyntaxError):\n" + "\n".join(problems)


def test_app_js_only_touches_element_ids_that_exist_in_index_html() -> None:
    """Regression: renamed/removed ids silently disabled Fire, Cast, and every sampling input."""
    html = _read("index.html")
    ids = set(re.findall(r'id="([^"]+)"', html))
    app = _read("app.js")

    referenced: set[str] = set()
    referenced.update(re.findall(r"""\$\(\s*["']#([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""byId\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""getElementById\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""setText\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""setHtml\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""\bnum\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""\bstr\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    referenced.update(re.findall(r"""\bchecked\(\s*["']([A-Za-z0-9_-]+)["']""", app))
    # Dynamic ids built from status payloads (`tab-${...}`) are allowed.
    referenced = {r for r in referenced if not r.startswith("tab-")}

    missing = sorted(r for r in referenced if r not in ids)
    assert not missing, f"app.js references element ids missing from index.html: {missing}"


def test_core_controls_are_bound_and_point_at_training_controls() -> None:
    """The Fire/Cast/Douse/Generate controls must exist AND be bound in app.js."""
    app = _read("app.js")
    for control in ("btn-init", "btn-fire", "btn-douse", "btn-gen", "btn-gen-stop"):
        assert f'byId("{control}")' in app or f'$("#{control}")' in app, f"{control} is never bound"
    # Sampling inputs must actually reach the RPC payload.
    for field in ("gen-tokens", "gen-temp", "gen-k", "gen-p", "gen-rep", "gen-seed", "gen-task"):
        assert f'"{field}"' in app, f"Voice input #{field} is not read by app.js"


def test_every_rpc_method_the_client_calls_is_dispatched_by_the_backend(tmp_path) -> None:
    app = _read("app.js")
    called = set(re.findall(r"""rpc\(\s*["']([a-zA-Z0-9_.]+)["']""", app))
    assert called, "no rpc() calls found — the contract test would be vacuous"

    backend_src = (Path(__file__).resolve().parent.parent / "ember" / "backend.py").read_text(encoding="utf-8")
    handled = set(re.findall(r'method\s*(?:==|in)\s*\(?\s*"([a-zA-Z0-9_.]+)"', backend_src))
    handled.update(re.findall(r'"([a-zA-Z0-9_.]+)"\s*,\s*"([a-zA-Z0-9_.]+)"\s*\)', backend_src))
    # `if method in ("file.commit", "file.resume")` style tuples
    for group in re.findall(r'method in \(([^)]*)\)', backend_src):
        handled.update(re.findall(r'"([a-zA-Z0-9_.]+)"', group))

    unknown = sorted(m for m in called if m not in handled)
    assert not unknown, f"app.js calls backend methods that ember.backend never dispatches: {unknown}"

    # And prove it at runtime: an unknown method must raise, a known one must not.
    be = EmberBackend(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unknown backend method"):
            be.dispatch("definitely.not.a.method")
        assert be.dispatch("ping")["pong"] is True
    finally:
        be.close()


def test_notify_calls_use_the_level_title_body_signature() -> None:
    """Regression: `notify(message, "ok")` inverted (level, title, body) and lost the message."""
    app = _read("app.js")
    bad = [m for m in re.findall(r'notify\(\s*"([^"]+)"', app) if m not in {"info", "success", "warn", "error"}]
    assert not bad, f"notify() called with a message as the level argument: {bad[:5]}"


def test_service_worker_caches_only_existing_files_and_bumps_the_cache_name() -> None:
    sw = _read("sw.js")
    cache_name = re.search(r'const CACHE = "([^"]+)"', sw)
    assert cache_name, "sw.js must declare a CACHE name"
    assert cache_name.group(1) != "ember-v2", (
        "sw.js still uses the ember-v2 cache: returning PWAs would be served deleted "
        "engine.js/tokenizer.js/brain.js copies from the old TensorFlow.js build."
    )
    shell = re.search(r"const SHELL = \[(.*?)\];", sw, re.S)
    assert shell, "sw.js must declare a SHELL precache list"
    entries = re.findall(r'"\./([^"]+)"', shell.group(1))
    assert entries, "SHELL must not be empty"
    missing = [e for e in entries if not (WEB_DIR / e).exists()]
    assert not missing, f"sw.js precaches files that do not exist: {missing}"
    for deleted in ("engine.js", "tokenizer.js", "brain.js", "store.js", "scrape.js", "teachers.js", "github.js"):
        assert deleted not in sw, f"sw.js still references deleted module {deleted}"
    assert "vendor/tf.min.js" not in sw, "sw.js must not precache the removed TensorFlow.js bundle"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to parse ES modules")
def test_web_modules_parse_as_es_modules_under_node() -> None:
    """`node --check` catches syntax errors without executing DOM-dependent code."""
    failures: list[str] = []
    for name in MODULES:
        src = WEB_DIR / name
        tmp = Path("/tmp") / f"ember-web-check-{name.replace('.js', '.mjs')}"
        tmp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(f"{name}: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'parse error'}")
        tmp.unlink(missing_ok=True)
    assert not failures, "web modules failed to parse:\n" + "\n".join(failures)
