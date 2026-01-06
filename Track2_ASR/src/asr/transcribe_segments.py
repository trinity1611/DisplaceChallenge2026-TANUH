from pathlib import Path
# from pydub import AudioSegment
import numpy as np
import torch
from transformers import AutoModel
from typing import Dict, Any
from utils.dataloader import ASRDataLoader
from utils.logger import LoggerSetup
import os
import soundfile as sf
from scipy.signal import resample_poly

logger = LoggerSetup.get_logger("asr_transcribe_segments")

class ASRSegmentPipeline:
    """
    ASR segmentation pipeline
    - Loads HF model and processor from asr_config (trust_remote_code for both)
    - Loads folders and RTTM info from common_config
    - Logs progress using LoggerSetup
    """
    def __init__(self, common_config: Dict[str, Any], asr_config: Dict[str, Any]):
        self.common_config = common_config
        self.asr_config = asr_config

        # Folders from common_config (use sane defaults if keys missing)
        self.audio_folder = Path(common_config.get("input_asr", {}).get("input_audio_folder", "data/Audio"))
        self.seg_out_folder = Path(common_config.get("output_asr", {}).get("seg_out_folder", "outputs/ASR/seg_asr_predictions"))
        self.fa_out_folder = Path(common_config.get("output_asr", {}).get("fa_out_folder", "outputs/ASR/fa_asr_predictions"))

        # Create folders if not exist
        self.seg_out_folder.mkdir(parents=True, exist_ok=True)
        self.fa_out_folder.mkdir(parents=True, exist_ok=True)

        # Load ASR model
        model_cfg = asr_config.get("model", {})
        self.model_id = model_cfg.get("model_id") or model_cfg.get("name")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading ASR model from: {self.model_id} on device {self.device}")
        try:
            self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
            self.model.eval()
            logger.info("ASR Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {str(e)}")
            raise

        # Initialize dataloader for RTTM parsing
        self.dataloader = ASRDataLoader(common_config)

    def transcribe_all(self, target_sr: int = 16000):
        """
        Iterate over RTTM files and transcribe segments + full audio
        """
        records = self.dataloader.load_records()
        logger.info(f"Found {len(records)} audio records for transcription")

        language_id = self.asr_config.get("language_id", "hi")
        decoding = self.asr_config.get("decoder", {}).get("cur_decoder", "rnnt")

        for rec in records:
            rec_id = rec["rec_id"]
            wav_file = self.audio_folder / f"{rec_id}.wav"
            logger.info(f'wav_file {wav_file}')

            if not wav_file.exists():
                logger.warning(f"Audio missing for {rec_id}, skipping...")
                continue

            rttm_segments = self.dataloader.parse_rttm(rec["rttm_file"])
            # ---- Load audio ----
            audio, sr = sf.read(wav_file, dtype="float32")
            # check if Mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            # Resample 
            if sr != target_sr:
                audio = resample_poly(audio, target_sr, sr)
                sr = target_sr

            full_transcript = ""
            logger.info(f"Processing record {rec_id} with {len(rttm_segments)} segments")
            # Segment-level transcription
            seg_out_path = self.seg_out_folder / f"{rec_id}_segment_transcription.txt"
            with seg_out_path.open("w", encoding="utf-8") as f_out:
                for i, seg in enumerate(rttm_segments, start=1):
                    # ---- Correct sample indexing ----
                    start = max(0, int(seg["start_time"] * sr) - int(0.1 * sr))
                    end   = min(len(audio), int(seg["end_time"] * sr) + int(0.1 * sr))

                    segment = audio[start:end]
                    seg_filename = f"{rec_id}_{i:04d}.wav"

                    if len(segment) == 0:
                        transcription_text = ""
                        continue

                    # ---- Convert to torch ----
                    waveform = torch.from_numpy(segment).unsqueeze(0).to(self.device)

                    # Pad if too short (minimum length for model compatibility)
                    min_length = 512
                    try:
                        if waveform.shape[1] < min_length:
                            padding = min_length - waveform.shape[1]
                            waveform = torch.nn.functional.pad(waveform, (0, padding))

                        # Transcribe using model directly
                        with torch.no_grad():
                            transcription = self.model(waveform, language_id, decoding)

                        transcription_text = transcription[0] if isinstance(transcription, (list, tuple)) else str(transcription)
                        full_transcript += transcription_text + " "
                        #logger.info(f"Segment {seg_filename} transcribed")
                    except Exception as e:
                        logger.error(f"Failed to transcribe {seg_filename}: {str(e)}")
                        transcription_text = ""
                    f_out.write(f"{rec_id}\t{seg_filename[:-4]}\t{seg['start_time']:.2f}\t"
                                f"{seg['end_time']:.2f}\t{seg['speaker_id']}\t{transcription_text}\n")

            # Full audio transcription
            fa_out_path = self.fa_out_folder / f"{rec_id}_fullaudio_transcription.txt"
            with fa_out_path.open("w", encoding="utf-8") as f:
                f.write(full_transcript.strip())
            logger.info(f"Saved full transcript for {rec_id} at {fa_out_path}")

def run_transcription(audio_folder: Path, rttm_folder: Path, seg_out_folder: Path, fa_out_folder: Path, model_config: Dict[str, Any]):
    common_config = {
        "input_asr": {"input_audio_folder": str(audio_folder), "input_gt": ""},
        "output_asr": {"seg_out_folder": str(seg_out_folder), "fa_out_folder": str(fa_out_folder)},
        "paths": {"diarization_output_dir": str(rttm_folder)}
    }
    pipeline = ASRSegmentPipeline(common_config, model_config)
    pipeline.transcribe_all()
