"""Static host for the on-device Ember foundry."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(title="Ember", version="0.2.0")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
