from pathlib import Path
from typing import List, Dict, Any
import re


class ASRDataLoader:
    """
    Loads ASR data for trancription and WER/CER evaluation.
    """
    def __init__(self, common_config: Dict[str, Any]):
        self.gt_folder = Path(common_config["input_asr"]["input_gt"])
        self.rttm_folder = Path(common_config["paths"]["diarization_output_dir"])
        self.seg_folder = Path(common_config["output_asr"]["seg_out_folder"])
        self.fa_folder = Path(common_config["output_asr"]["fa_out_folder"])

    @staticmethod
    def parse_rttm(rttm_path: Path) -> List[Dict[str, Any]]:
        """Parse RTTM into segment list"""
        segments = []
        with rttm_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip() or not line.startswith("SPEAKER"):
                    continue
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                segments.append({
                    "start_time": start_time,
                    "end_time": start_time + duration,
                    "speaker_id": speaker_id
                })
        return segments

    @staticmethod
    def parse_seg_pred(seg_pred_path: Path) -> List[Dict[str, Any]]:
        """Segment-level ASR predictions into segment list"""
        segments = []
        with seg_pred_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 6:
                    continue
                sess, utt, start, end, spk, text = parts
                segments.append({
                    "session_id": sess,
                    "speaker": spk,
                    "start_time": float(start),
                    "end_time": float(end),
                    "words": text
                })
        return segments

    @staticmethod
    def parse_gt_file(gt_file: Path) -> List[Dict[str, Any]]:
        """Segment-level ground truth into segment list"""
        segments = []
        with gt_file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 7:
                    continue
                sess, utt, spk, spk_name, start, end, text = parts
                segments.append({
                    "session_id": sess,
                    "speaker": spk_name,
                    "start_time": float(start),
                    "end_time": float(end),
                    "words": text
                })
        return segments

   
    @staticmethod
    def build_full_transcript_from_segments(gt_file: Path) -> str:
        """Build full transcript from segmented GT file"""
        segments = []
        with gt_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue

                text = parts[-1].strip()
                if text:
                    segments.append(text)

        if not segments:
            return ""

        return " ".join(text for text in segments)

    def load_records(self) -> List[Dict[str, Any]]:
        """
        Prepare data records for transcription / metrics
        """
        records = []
        for rttm_file in self.rttm_folder.glob("*.rttm"):
            rec_id = rttm_file.stem
            pred_file = self.fa_folder / f"{rec_id}_fullaudio_transcription.txt"
            gt_file = self.gt_folder / f"{rec_id}.txt"
            records.append({
                "rec_id": rec_id,
                "gt_file": gt_file,
                "pred_file": pred_file,
                "rttm_file": rttm_file
            })
        return records