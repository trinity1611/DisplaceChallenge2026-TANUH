"""
DISPLACE MedAI – ASR Transcription Service (Track 2)
======================================================
Wraps the IndicConformer-600M model to transcribe audio segments
identified by the diarization stage.
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly

from backend.app.config import settings

logger = logging.getLogger("displace.transcription")


class TranscriptionService:
    """ASR transcription using IndicConformer (Track 2)."""

    def __init__(self):
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the IndicConformer ASR model."""
        if self._loaded:
            logger.info("ASR model already loaded, skipping.")
            return

        model_id = settings.asr_model_id
        device = settings.device

        logger.info(f"Loading ASR model: {model_id} on {device}")

        from transformers import AutoModel

        # Set HF token for gated models
        hf_token = settings.hf_token or None

        self._model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        ).to(device)
        self._model.eval()

        self._loaded = True
        logger.info("ASR model loaded successfully")

    def unload_model(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("ASR model unloaded")

    def run(
        self,
        audio_path: Path,
        diarization_segments: List[Dict[str, Any]],
        language_id: str = "hi",
        target_sr: int = 16000,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using diarization segments.

        Args:
            audio_path: Path to the .wav file
            diarization_segments: List of {start_time, end_time, speaker_id}
            language_id: Language code ("hi" for Hindi, "kn" for Kannada)

        Returns:
            {
                "segments": [{"segment_id": int, "start_time": float,
                              "end_time": float, "speaker_id": str,
                              "text": str}, ...],
                "full_transcript": str,
                "elapsed_s": float,
            }
        """
        start_time_wall = time.time()

        if not self._loaded:
            self.load_model()

        logger.info(f"Transcribing {audio_path.name} with {len(diarization_segments)} segments")

        # Load full audio
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr).astype(np.float32)
            sr = target_sr

        device = settings.device
        decoding = "rnnt"
        transcript_segments = []
        full_transcript_parts = []

        for i, seg in enumerate(diarization_segments, start=1):
            # Extract audio segment with a small buffer
            start_sample = max(0, int(seg["start_time"] * sr) - int(0.1 * sr))
            end_sample = min(len(audio), int(seg["end_time"] * sr) + int(0.1 * sr))
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            # Convert to torch tensor
            waveform = torch.from_numpy(segment_audio).unsqueeze(0).to(device)

            # Pad if too short
            min_length = 512
            if waveform.shape[1] < min_length:
                padding = min_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            try:
                with torch.no_grad():
                    transcription = self._model(waveform, language_id, decoding)

                text = (
                    transcription[0]
                    if isinstance(transcription, (list, tuple))
                    else str(transcription)
                )
                text = text.strip()
            except Exception as e:
                logger.error(f"Failed to transcribe segment {i}: {e}")
                text = ""

            if text:
                full_transcript_parts.append(text)

            transcript_segments.append({
                "segment_id": i,
                "start_time": round(seg["start_time"], 3),
                "end_time": round(seg["end_time"], 3),
                "speaker_id": seg["speaker_id"],
                "text": text,
            })

        full_transcript = " ".join(full_transcript_parts).strip()
        elapsed = time.time() - start_time_wall

        logger.info(
            f"Transcription complete: {len(transcript_segments)} segments, "
            f"{len(full_transcript)} chars, {elapsed:.1f}s"
        )

        return {
            "segments": transcript_segments,
            "full_transcript": full_transcript,
            "elapsed_s": elapsed,
        }


# Singleton instance
transcription_service = TranscriptionService()
