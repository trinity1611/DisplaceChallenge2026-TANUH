"""
DISPLACE MedAI – Pydantic Schemas
====================================
Request/response models for the API.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# ── Upload ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    message: str = "File uploaded successfully"


# ── Process ──────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    language: str = "hi"  # hi, kn, auto


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── Job Status ───────────────────────────────────────────────────────

class JobStatus(BaseModel):
    id: str
    filename: str
    original_filename: str
    language: str
    status: str
    progress: int
    stage_message: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Diarization Segment ─────────────────────────────────────────────

class DiarizationSegment(BaseModel):
    start_time: float
    end_time: float
    speaker_id: str
    duration: float


# ── Transcript Segment ──────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    speaker_id: str
    text: str


# ── Full Results ─────────────────────────────────────────────────────

class PipelineResults(BaseModel):
    job_id: str
    status: str

    # Track 1
    num_speakers: int = 0
    diarization: List[DiarizationSegment] = []

    # Track 2
    transcript: List[TranscriptSegment] = []
    full_transcript: str = ""

    # Track 3
    topics: str = ""
    topics_list: List[str] = []

    # Track 4
    summary: str = ""

    # Timing
    diarization_time_s: float = 0.0
    asr_time_s: float = 0.0
    topic_time_s: float = 0.0
    summary_time_s: float = 0.0
    total_time_s: float = 0.0


# ── Health Check ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


# ── Models Status ────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    track: str
    loaded: bool
    model_id: str


class ModelsStatusResponse(BaseModel):
    models: List[ModelInfo]
    device: str
