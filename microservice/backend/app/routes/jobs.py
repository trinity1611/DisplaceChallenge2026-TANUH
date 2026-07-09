"""
DISPLACE MedAI – Job Routes
===============================
Endpoints for starting processing, checking status,
and retrieving results.
"""

import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import Job, Result
from backend.app.schemas import (
    ProcessRequest,
    ProcessResponse,
    JobStatus,
    PipelineResults,
    DiarizationSegment,
    TranscriptSegment,
)
from backend.app.services.pipeline import run_pipeline

logger = logging.getLogger("displace.routes.jobs")

router = APIRouter(prefix="/api", tags=["Jobs"])

# Track active pipeline threads to prevent concurrent GPU jobs
_active_pipeline_lock = threading.Lock()
_active_job_id: Optional[str] = None


@router.post("/process/{job_id}", response_model=ProcessResponse)
async def start_processing(job_id: str, request: ProcessRequest = None):
    """
    Start the 4-stage ML pipeline for an uploaded audio file.

    The pipeline runs in a background thread. Poll GET /api/jobs/{job_id}
    for real-time status updates.
    """
    global _active_job_id

    if request is None:
        request = ProcessRequest()

    # Check the job exists
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in ("UPLOADED", "FAILED"):
            raise HTTPException(
                status_code=409,
                detail=f"Job is already {job.status}. Cannot restart.",
            )

        # Check if another pipeline is running
        with _active_pipeline_lock:
            if _active_job_id is not None:
                raise HTTPException(
                    status_code=429,
                    detail=f"Another pipeline is currently running (job: {_active_job_id}). "
                    "Please wait for it to complete.",
                )
            _active_job_id = job_id

        # Update job
        job.status = "QUEUED"
        job.progress = 0
        job.language = request.language
        job.stage_message = "Pipeline queued, starting soon..."
        job.error_message = None
        db.commit()

        # Build audio path
        audio_path = str(settings.uploads_dir / job.filename)

    finally:
        db.close()

    # Launch pipeline in background thread
    def _run_and_cleanup():
        global _active_job_id
        try:
            run_pipeline(job_id, audio_path, request.language)
        finally:
            with _active_pipeline_lock:
                _active_job_id = None

    thread = threading.Thread(target=_run_and_cleanup, daemon=True)
    thread.start()

    logger.info(f"Pipeline started for job {job_id} (language: {request.language})")

    return ProcessResponse(
        job_id=job_id,
        status="QUEUED",
        message="Pipeline started. Poll GET /api/jobs/{job_id} for status.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the current status and progress of a pipeline job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobStatus(
            id=job.id,
            filename=job.filename,
            original_filename=job.original_filename,
            language=job.language,
            status=job.status,
            progress=job.progress,
            stage_message=job.stage_message,
            error_message=job.error_message,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )
    finally:
        db.close()


@router.get("/jobs/{job_id}/results", response_model=PipelineResults)
async def get_job_results(job_id: str):
    """
    Get the full pipeline results for a completed job.

    Returns diarization segments, transcript, topics, and summary.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")



        result = db.query(Result).filter(Result.job_id == job_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="Results not found")

        # Parse JSON fields
        diarization = [
            DiarizationSegment(**seg)
            for seg in json.loads(result.diarization_json)
        ]
        transcript = [
            TranscriptSegment(**seg)
            for seg in json.loads(result.transcript_json)
        ]

        topics_str = result.topics or ""
        topics_list = [
            t.strip() for t in topics_str.split(",") if t.strip()
        ]

        return PipelineResults(
            job_id=job_id,
            status=job.status,
            num_speakers=result.num_speakers,
            diarization=diarization,
            transcript=transcript,
            full_transcript=result.full_transcript,
            topics=result.topics,
            topics_list=topics_list,
            summary=result.summary,
            diarization_time_s=result.diarization_time_s,
            asr_time_s=result.asr_time_s,
            topic_time_s=result.topic_time_s,
            summary_time_s=result.summary_time_s,
            total_time_s=result.total_time_s,
        )
    finally:
        db.close()
