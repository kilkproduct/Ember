"""Self-contained HTTP + JSON-RPC server for desktop/CLI runs (`python3 -m ember.server`).

Requires zero FastAPI/Pydantic Rust wheels so it works anywhere Python + NumPy run.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .backend import EmberBackend


def create_server(host: str = "0.0.0.0", port: int = 8000, data_dir: str | Path = "data") -> ThreadingHTTPServer:
    root = Path(__file__).resolve().parent.parent
    web_dir = root / "web"
    backend = EmberBackend(data_dir=root / data_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quiet health polls
            return

        def _send_json(self, code: int, obj: object) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/rpc":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    req = json.loads(body)
                    res = backend.dispatch(str(req.get("method") or ""), req.get("params") or {})
                    self._send_json(200, {"ok": True, "result": res})
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                return
            self._send_json(404, {"ok": False, "error": "Not found"})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._send_json(200, {"ok": True, "result": backend.full_status()})
                return
            rel = parsed.path.lstrip("/") or "index.html"
            target = (web_dir / rel).resolve()
            if not str(target).startswith(str(web_dir.resolve())) or not target.is_file():
                self.send_error(404)
                return
            ctype, _ = mimetypes.guess_type(str(target))
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ember Python + Native C++ server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    srv = create_server(args.host, args.port, args.data_dir)
    print(f"Ember v3 server listening on http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
