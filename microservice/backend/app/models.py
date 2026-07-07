"""
DISPLACE MedAI – Database Models
===================================
SQLAlchemy ORM models for tracking jobs and storing results.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from backend.app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


def _generate_uuid():
    return str(uuid.uuid4())


class Job(Base):
    """Represents a single audio processing pipeline job."""

    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_generate_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    language = Column(String, default="hi")  # hi, kn, auto
    status = Column(String, default="UPLOADED")
    # Status values: UPLOADED, DIARIZING, TRANSCRIBING, TOPIC_EXTRACTION,
    #                SUMMARIZING, COMPLETED, FAILED
    progress = Column(Integer, default=0)  # 0-100
    stage_message = Column(String, default="Waiting to start...")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationship to results
    result = relationship("Result", back_populates="job", uselist=False)


class Result(Base):
    """Stores the pipeline output for a completed job."""

    __tablename__ = "results"

    id = Column(String, primary_key=True, default=_generate_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, unique=True)

    # Track 1 – Speaker Diarization
    num_speakers = Column(Integer, default=0)
    diarization_json = Column(Text, default="[]")
    # JSON array of {start_time, end_time, speaker_id, duration}

    # Track 2 – ASR Transcript
    transcript_json = Column(Text, default="[]")
    # JSON array of {segment_id, start_time, end_time, speaker_id, text}
    full_transcript = Column(Text, default="")

    # Track 3 – Topic Identification
    topics = Column(Text, default="")
    # Comma-separated medical topics

    # Track 4 – Dialogue Summarization
    summary = Column(Text, default="")

    # Timing
    diarization_time_s = Column(Float, default=0.0)
    asr_time_s = Column(Float, default=0.0)
    topic_time_s = Column(Float, default=0.0)
    summary_time_s = Column(Float, default=0.0)
    total_time_s = Column(Float, default=0.0)

    # Relationship
    job = relationship("Job", back_populates="result")
