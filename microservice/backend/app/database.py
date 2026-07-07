"""
DISPLACE MedAI – Database Setup
=================================
SQLite database with SQLAlchemy for tracking pipeline jobs.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
