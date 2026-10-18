"""Static host for the on-device Ember foundry."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(title="Ember", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")
