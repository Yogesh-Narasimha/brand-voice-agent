"""FastAPI app: REST API under /api plus the static web UI at /."""
import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import agent, brand_kits, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("brand-voice-agent")

app = FastAPI(title="Brand Voice Content Agent", version="1.0.0")
STATIC = Path(__file__).parent / "static"


class GenerateRequest(BaseModel):
    brand_kit_id: str
    brief: str = Field(min_length=10, max_length=2000)
    channel: Literal["email", "linkedin", "instagram", "ad", "product"] = "linkedin"


@app.get("/api/health")
def health():
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "model": config.CHAT_MODEL}


@app.post("/api/brand-kits", status_code=201)
async def create_brand_kit(name: str = Form(..., min_length=1, max_length=80), files: list[UploadFile] = File(...)):
    loaded = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"{f.filename}: only .txt, .md and .pdf files are allowed")
        data = await f.read()
        if len(data) > config.MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} is larger than {config.MAX_FILE_MB} MB")
        loaded.append((f.filename, data))
    try:
        meta = brand_kits.create_kit(name, loaded)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("Created brand kit %s (%d chunks)", meta["id"], meta["chunks"])
    return meta


@app.get("/api/brand-kits")
def list_brand_kits():
    return brand_kits.list_kits()


@app.get("/api/brand-kits/{kit_id}")
def get_brand_kit(kit_id: str):
    try:
        return brand_kits.get_kit_meta(kit_id)
    except KeyError:
        raise HTTPException(404, "Brand kit not found")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        brand_kits.get_kit_meta(req.brand_kit_id)
    except KeyError:
        raise HTTPException(404, "Brand kit not found")
    start = time.perf_counter()
    try:
        result = agent.generate(req.brand_kit_id, req.brief, req.channel)
    except Exception as e:  # LLM/API errors reach the user as a clean 502
        log.exception("Generation failed")
        raise HTTPException(502, f"Generation failed: {e}")
    result["latency_ms"] = round((time.perf_counter() - start) * 1000)
    log.info("Generated %s content for kit %s: score=%s revisions=%s",
             req.channel, req.brand_kit_id, result["review"]["score"], result["revisions"])
    return result


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
