"""
DISPLACE MedAI – FHIR Routes
================================
Endpoints for FHIR R4 Bundle extraction from completed job summaries.
"""

import json
import logging

from fastapi import APIRouter, HTTPException

from backend.app.database import SessionLocal
from backend.app.models import Job, Result
from backend.app.schemas import FHIRResponse
from backend.app.services.fhir_extraction import fhir_extraction_service

logger = logging.getLogger("displace.routes.fhir")

router = APIRouter(prefix="/api", tags=["FHIR"])


@router.post("/fhir/{job_id}", response_model=FHIRResponse)
async def extract_fhir(job_id: str):
    """
    Extract a FHIR R4 Bundle from a completed job's summary.

    Requires the job to have status COMPLETED with a non-empty summary.
    Calls MedGemma-27B-IT via vLLM to generate the FHIR bundle.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status != "COMPLETED":
            raise HTTPException(
                status_code=409,
                detail=f"Job status is '{job.status}'. FHIR extraction requires COMPLETED status.",
            )

        result = db.query(Result).filter(Result.job_id == job_id).first()
        if not result or not result.summary:
            raise HTTPException(
                status_code=404,
                detail="No summary found for this job. Run the pipeline first.",
            )

        summary = result.summary
    finally:
        db.close()

    # Run FHIR extraction
    logger.info(f"Starting FHIR extraction for job {job_id}")
    fhir_result = fhir_extraction_service.run(summary)

    # Store result in database
    db = SessionLocal()
    try:
        result = db.query(Result).filter(Result.job_id == job_id).first()
        if result:
            result.fhir_bundle = fhir_result["fhir_json"]
            result.fhir_time_s = round(fhir_result["elapsed_s"], 2)
            db.commit()
    finally:
        db.close()

    return FHIRResponse(
        job_id=job_id,
        fhir_bundle=fhir_result["fhir_bundle"],
        fhir_json=fhir_result["fhir_json"],
        valid=fhir_result["valid"],
        validation_errors=fhir_result["validation_errors"],
        elapsed_s=fhir_result["elapsed_s"],
    )


@router.get("/fhir/{job_id}", response_model=FHIRResponse)
async def get_fhir(job_id: str):
    """
    Get the previously extracted FHIR R4 Bundle for a job.

    Returns the stored bundle without re-running extraction.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        result = db.query(Result).filter(Result.job_id == job_id).first()
        if not result or not result.fhir_bundle:
            raise HTTPException(
                status_code=404,
                detail="No FHIR bundle found. POST /api/fhir/{job_id} to extract one.",
            )

        # Parse stored JSON
        try:
            bundle_dict = json.loads(result.fhir_bundle)
        except (json.JSONDecodeError, TypeError):
            bundle_dict = None

        return FHIRResponse(
            job_id=job_id,
            fhir_bundle=bundle_dict,
            fhir_json=result.fhir_bundle,
            valid=bundle_dict is not None,
            validation_errors=[],
            elapsed_s=result.fhir_time_s or 0.0,
        )
    finally:
        db.close()
