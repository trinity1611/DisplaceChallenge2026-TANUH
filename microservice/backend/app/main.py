"""
DISPLACE MedAI – FastAPI Application
=======================================
Entry point for the microservice. Mounts routes and serves
the frontend static files.
"""

import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database import engine
from backend.app import models
from backend.app.routes import audio, jobs
from backend.app.schemas import HealthResponse, ModelsStatusResponse, ModelInfo

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("displace.main")

# ── Create database tables ──────────────────────────────────────────

models.Base.metadata.create_all(bind=engine)

# ── Application ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "DISPLACE MedAI – A GPU-accelerated microservice for processing "
        "healthcare conversations. Upload raw audio → get speaker diarization, "
        "ASR transcripts, medical topic extraction, and dialogue summarization."
    ),
)

# CORS – allow frontend from same origin or dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ────────────────────────────────────────────────

app.include_router(audio.router)
app.include_router(jobs.router)


# ── Health check ─────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check with GPU info."""
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
        )

    return HealthResponse(
        status="healthy",
        service=settings.app_title,
        version=settings.app_version,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


# ── Models status ────────────────────────────────────────────────────

@app.get("/api/models/status", response_model=ModelsStatusResponse, tags=["System"])
async def models_status():
    """Check which ML models are currently loaded."""
    from backend.app.services.diarization import diarization_service
    from backend.app.services.transcription import transcription_service
    from backend.app.services.topic_extraction import topic_extraction_service
    from backend.app.services.summarization import summarization_service

    return ModelsStatusResponse(
        device=settings.device,
        models=[
            ModelInfo(
                name="DiariZen (Speaker Diarization)",
                track="Track 1 – SD",
                loaded=diarization_service.is_loaded,
                model_id="BUT-FIT/diarizen-wavlm_base_s80_md",
            ),
            ModelInfo(
                name="IndicConformer (ASR)",
                track="Track 2 – ASR",
                loaded=transcription_service.is_loaded,
                model_id=settings.asr_model_id,
            ),
            ModelInfo(
                name="Qwen 2.5-3B (Topic Extraction)",
                track="Track 3 – TI",
                loaded=topic_extraction_service.is_loaded,
                model_id=settings.qwen_model_id,
            ),
            ModelInfo(
                name="LLaMA 3.2-3B (Summarization)",
                track="Track 4 – DS",
                loaded=summarization_service.is_loaded,
                model_id=settings.llama_model_id,
            ),
        ],
    )


# ── Mount frontend ──────────────────────────────────────────────────

_frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_path.exists():
    app.mount(
        "/", StaticFiles(directory=str(_frontend_path), html=True), name="frontend"
    )
    logger.info(f"Frontend mounted from: {_frontend_path}")
else:
    logger.warning(f"Frontend directory not found at: {_frontend_path}")
