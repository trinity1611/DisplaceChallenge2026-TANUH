"""
DISPLACE MedAI – Audio Routes
================================
Endpoints for uploading audio files.
"""

import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import Job
from backend.app.schemas import UploadResponse

logger = logging.getLogger("displace.routes.audio")

router = APIRouter(prefix="/api", tags=["Audio"])


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload a .wav audio file for processing.

    Returns a job_id that can be used to start processing
    and poll for status.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    # Generate unique filename
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{ext}"
    save_path = settings.uploads_dir / safe_filename

    # Save file to disk
    with open(save_path, "wb") as f:
        f.write(content)

    # Create job record in database
    db = SessionLocal()
    try:
        job = Job(
            id=job_id,
            filename=safe_filename,
            original_filename=file.filename,
            status="UPLOADED",
            progress=0,
            stage_message="File uploaded, ready to process",
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    logger.info(f"Uploaded {file.filename} as {safe_filename} (job: {job_id})")

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        message="File uploaded successfully. Use POST /api/process/{job_id} to start processing.",
    )
