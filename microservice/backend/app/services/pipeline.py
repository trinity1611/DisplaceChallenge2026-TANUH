"""
DISPLACE MedAI – Pipeline Orchestrator
=========================================
Chains all 4 tracks sequentially:
  Track 1 (SD) → Track 2 (ASR) → Track 3 (TI) + Track 4 (DS)

Runs as a background task, updating job status in the database
so the frontend can poll for progress.
"""

import json
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models import Job, Result
from backend.app.services.diarization import diarization_service
from backend.app.services.transcription import transcription_service
from backend.app.services.topic_extraction import topic_extraction_service
from backend.app.services.summarization import summarization_service

logger = logging.getLogger("displace.pipeline")


def _update_job(job_id: str, **kwargs) -> None:
    """Helper to update job status in the database."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            for key, value in kwargs.items():
                setattr(job, key, value)
            db.commit()
    finally:
        db.close()


def run_pipeline(job_id: str, audio_path: str, language: str = "hi") -> None:
    """
    Execute the full 4-stage ML pipeline for a given job.

    This function runs synchronously in a background thread.
    It updates the Job record at each stage so the frontend
    can display real-time progress.
    """
    audio_file = Path(audio_path)
    total_start = time.time()

    logger.info(f"═══ Pipeline started for job {job_id} ═══")
    logger.info(f"Audio: {audio_file.name}, Language: {language}")

    try:
        # ──────────────────────────────────────────────────────────
        # STAGE 1: Speaker Diarization (Track 1)
        # ──────────────────────────────────────────────────────────
        _update_job(
            job_id,
            status="DIARIZING",
            progress=5,
            stage_message="Loading diarization model...",
        )

        diarization_result = diarization_service.run(audio_file)
        diar_segments = diarization_result["segments"]
        num_speakers = diarization_result["num_speakers"]

        _update_job(
            job_id,
            progress=25,
            stage_message=f"Diarization complete – {num_speakers} speakers, {len(diar_segments)} segments",
        )

        logger.info(
            f"[Stage 1] Diarization: {num_speakers} speakers, "
            f"{len(diar_segments)} segments, {diarization_result['elapsed_s']:.1f}s"
        )
        
        # Unload Diarization model to free VRAM for ASR
        diarization_service.unload_model()

        # ──────────────────────────────────────────────────────────
        # STAGE 2: ASR Transcription (Track 2)
        # ──────────────────────────────────────────────────────────
        _update_job(
            job_id,
            status="TRANSCRIBING",
            progress=30,
            stage_message="Loading ASR model & transcribing...",
        )

        asr_result = transcription_service.run(
            audio_path=audio_file,
            diarization_segments=diar_segments,
            language_id=language,
        )
        transcript_segments = asr_result["segments"]
        full_transcript = asr_result["full_transcript"]

        _update_job(
            job_id,
            progress=55,
            stage_message=f"Transcription complete – {len(transcript_segments)} segments",
        )

        logger.info(
            f"[Stage 2] ASR: {len(transcript_segments)} segments, "
            f"{len(full_transcript)} chars, {asr_result['elapsed_s']:.1f}s"
        )
        
        # Unload ASR model to free VRAM for Topic Extraction (Qwen)
        transcription_service.unload_model()

        # ──────────────────────────────────────────────────────────
        # STAGE 3: Topic Identification (Track 3)
        # ──────────────────────────────────────────────────────────
        _update_job(
            job_id,
            status="TOPIC_EXTRACTION",
            progress=60,
            stage_message="Extracting medical topics...",
        )

        topic_result = topic_extraction_service.run(full_transcript)

        _update_job(
            job_id,
            progress=75,
            stage_message=f"Topics extracted: {topic_result['topics']}",
        )

        logger.info(
            f"[Stage 3] Topics: {topic_result['topics']} ({topic_result['elapsed_s']:.1f}s)"
        )

        # Unload Qwen to free VRAM for LLaMA
        topic_extraction_service.unload_model()

        # ──────────────────────────────────────────────────────────
        # STAGE 4: Dialogue Summarization (Track 4)
        # ──────────────────────────────────────────────────────────
        _update_job(
            job_id,
            status="SUMMARIZING",
            progress=80,
            stage_message="Generating dialogue summary...",
        )

        summary_result = summarization_service.run(full_transcript)

        _update_job(
            job_id,
            progress=95,
            stage_message="Saving results...",
        )

        logger.info(
            f"[Stage 4] Summary: {len(summary_result['summary'])} chars "
            f"({summary_result['elapsed_s']:.1f}s)"
        )

        # Unload LLaMA to free VRAM
        summarization_service.unload_model()

        # ──────────────────────────────────────────────────────────
        # SAVE RESULTS
        # ──────────────────────────────────────────────────────────
        total_elapsed = time.time() - total_start

        db = SessionLocal()
        try:
            result = Result(
                job_id=job_id,
                num_speakers=num_speakers,
                diarization_json=json.dumps(diar_segments),
                transcript_json=json.dumps(transcript_segments),
                full_transcript=full_transcript,
                topics=topic_result["topics"],
                summary=summary_result["summary"],
                diarization_time_s=round(diarization_result["elapsed_s"], 2),
                asr_time_s=round(asr_result["elapsed_s"], 2),
                topic_time_s=round(topic_result["elapsed_s"], 2),
                summary_time_s=round(summary_result["elapsed_s"], 2),
                total_time_s=round(total_elapsed, 2),
            )
            db.add(result)

            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "COMPLETED"
                job.progress = 100
                job.stage_message = f"Pipeline complete in {total_elapsed:.1f}s"
                job.completed_at = datetime.now(timezone.utc)

            db.commit()
        finally:
            db.close()

        logger.info(
            f"═══ Pipeline completed for job {job_id} in {total_elapsed:.1f}s ═══"
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Pipeline failed for job {job_id}: {error_msg}")
        logger.error(traceback.format_exc())

        _update_job(
            job_id,
            status="FAILED",
            stage_message="Pipeline failed",
            error_message=error_msg,
        )
