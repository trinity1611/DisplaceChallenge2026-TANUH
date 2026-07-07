"""
DISPLACE MedAI – Application Configuration
============================================
Central configuration loaded from environment variables / .env file.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


# Resolve paths relative to the microservice root
_MICROSERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
_BASELINES_ROOT = _MICROSERVICE_ROOT.parent  # DISPLACE-2026-Baselines/


class Settings(BaseSettings):
    """Central settings for the DISPLACE MedAI microservice."""

    # ── App ──────────────────────────────────────────────────────────
    app_title: str = "DISPLACE MedAI"
    app_version: str = "1.0.0"
    debug: bool = True

    # ── Server ───────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # ── HuggingFace ──────────────────────────────────────────────────
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # ── Device ───────────────────────────────────────────────────────
    device: str = Field(default="cuda", alias="DEVICE")

    # ── Model IDs ────────────────────────────────────────────────────
    asr_model_id: str = Field(
        default="ai4bharat/indic-conformer-600m-multilingual",
        alias="ASR_MODEL_ID",
    )
    qwen_model_id: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        alias="QWEN_MODEL_ID",
    )
    llama_model_id: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        alias="LLAMA_MODEL_ID",
    )

    # ── Upload ───────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=100, alias="MAX_UPLOAD_SIZE_MB")

    # ── Paths (computed, not from .env) ──────────────────────────────
    @property
    def microservice_root(self) -> Path:
        return _MICROSERVICE_ROOT

    @property
    def baselines_root(self) -> Path:
        return _BASELINES_ROOT

    @property
    def track1_dir(self) -> Path:
        return _BASELINES_ROOT / "Track1_SD"

    @property
    def track2_dir(self) -> Path:
        return _BASELINES_ROOT / "Track2_ASR"

    @property
    def track3_dir(self) -> Path:
        return _BASELINES_ROOT / "Track3_TI"

    @property
    def track4_dir(self) -> Path:
        return _BASELINES_ROOT / "Track4_DS"

    @property
    def track1_config_path(self) -> Path:
        return self.track1_dir / "config.toml"

    @property
    def uploads_dir(self) -> Path:
        d = _MICROSERVICE_ROOT / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> Path:
        d = _MICROSERVICE_ROOT / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self) -> Path:
        return _MICROSERVICE_ROOT / "displace_medai.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    class Config:
        env_file = str(_MICROSERVICE_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
