from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json
import jiwer
import re

try:
    import meeteval
except ImportError:
    meeteval = None

from utils.dataloader import ASRDataLoader
from utils.helpers import ConfigLoader
from utils.logger import LoggerSetup

COLLAR_INTERVAL = 3  # seconds collar for TCPWER

logger = LoggerSetup.get_logger("asr_metrics")


# MeetEval utilities
def normalizer(seg):
    seg['words'] = seg['words'].lower()
    seg['words'] = re.sub(r'[.?!,।]', '', seg['words'])
    seg['words'] = re.sub(r'<[^>]*>', '', seg['words'])
    return seg

def cpwer_to_dict(cpwer):
    return {
        "wer": cpwer.error_rate,
        "sub": cpwer.substitutions,
        "del": cpwer.deletions,
        "ins": cpwer.insertions,
        "missed_speaker": cpwer.missed_speaker,
        "falarm_speaker": cpwer.falarm_speaker,
        "scored_speaker": cpwer.scored_speaker,
        "assignment": list(cpwer.assignment)
    }

# Compute TCPWER per record - Changed type hint to "Any" to avoid None Type attribute errors
def compute_tcpwer_per_record(gt_segments: List[Dict[str, Any]], pred_segments: List[Dict[str, Any]]) -> Any:
    if meeteval is None:
        return None
    ref = meeteval.io.asseglst(gt_segments)
    hyp = meeteval.io.asseglst(pred_segments)
    cpwer = meeteval.wer.tcpwer(ref, hyp, collar=COLLAR_INTERVAL, normalizer=normalizer)
    return cpwer

class TranscriptionMetrics:
    """
    Compute CER and WER between full audio predictions and from segmented ground truth.
    """

    @staticmethod
    def calculate_wer_cer(common_config: Dict[str, Any], output_file: str = None) -> Dict[str, Any]:
        # Initialize dataloader
        dataloader = ASRDataLoader(common_config)
        logger.info("Loading data for WER/CER computation...")

        # Get predicted transcripts (full audio) and GT segments
        pred_folder = Path(common_config["output_asr"]["fa_out_folder"])
        gt_folder = Path(common_config["input_asr"]["input_gt"])

        pred_files = sorted(pred_folder.glob("*.txt"))
        logger.info(f"Found {len(pred_files)} predicted transcripts in {pred_folder}")

        wers: List[float] = []
        cers: List[float] = []
        file_results = []

        for pred_path in pred_files:
            rec_id = pred_path.stem.replace("_fullaudio_transcription", "")
            logger.info(f"rec_id {rec_id}")
            gt_file = gt_folder / f"{rec_id}.txt"
            logger.info(f"gt_file {gt_file}")

            if not gt_file.exists():
                logger.warning(f"Segmented GT missing for {rec_id}, skipping...")
                continue

            # Build full GT transcript from segments
            gt_text = dataloader.build_full_transcript_from_segments(gt_file)
            pred_text = pred_path.read_text(encoding="utf-8").strip()

            cer = jiwer.cer(gt_text, pred_text)
            wer = jiwer.wer(gt_text, pred_text)

            wers.append(wer)
            cers.append(cer)

            file_results.append({
                "rec_id": rec_id,
                "reference": gt_text,
                "prediction": pred_text,
                "cer": cer,
                "wer": wer
            })

            logger.info(f"{rec_id}: CER={cer:.3f}, WER={wer:.3f}")

        avg_cer = sum(cers) / len(cers) if cers else 0.0
        avg_wer = sum(wers) / len(wers) if wers else 0.0

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "records processed" : len(wers), 
            "average_cer": avg_cer,
            "average_wer": avg_wer,
            "files": file_results
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metrics to {output_path}")

        logger.info(f"Average CER: {avg_cer:.3f}, WER: {avg_wer:.3f}")
        return results
    
    # TCPWER calculation for segments
    @staticmethod
    def compute_tcpwer(common_config: Dict[str, Any], output_file: str = None, eval_speaker: bool = False):
        """
        Compute TCPWER across all records using ASRDataLoader
        """
        if meeteval is None:
            logger.warning("meeteval is not installed. Skipping TCPWER calculation step entirely.")
            return {"status": "skipped", "reason": "meeteval not installed"}

        # Initialize dataloader
        dataloader = ASRDataLoader(common_config)
        logger.info("Loading data for tcpWER computation...")

        # Get predicted transcripts (full audio) and GT segments
        pred_folder = Path(common_config["output_asr"]["seg_out_folder"])
        gt_folder = Path(common_config["input_asr"]["input_gt"])

        pred_files = sorted(pred_folder.glob("*.txt"))
        logger.info(f"Found {len(pred_files)} predicted transcripts in {pred_folder}")

        individual_results = {}
        all_wers = []

        for pred_path in pred_files:
            rec_id = pred_path.stem.replace("_segment_transcription", "")
            logger.info(f"rec_id {rec_id}")
            gt_file = gt_folder / f"{rec_id}.txt"
            logger.info(f"gt_file {gt_file}")

            if not gt_file.exists():
                logger.warning(f"Segmented GT missing for {rec_id}, skipping...")
                continue

            # Load segments
            gt_segments = dataloader.parse_gt_file(gt_file)
            pred_segments = dataloader.parse_seg_pred(pred_path)

            if not pred_segments:
                print(f"[WARN] Missing predictions for {rec_id}, skipping...")
                continue

            cpwer = compute_tcpwer_per_record(gt_segments, pred_segments)
            combined = meeteval.wer.combine_error_rates(cpwer)

            individual_results[rec_id] = cpwer_to_dict(combined)
            all_wers.append(combined.error_rate)

            logger.info(f"{rec_id}: TCPWER = {combined.error_rate:.3f}")

        avg_tcpwer = sum(all_wers) / len(all_wers) if all_wers else 0.0
        logger.info(f"Average TCPWER across all records: {avg_tcpwer:.3f}")

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "records processed" : len(all_wers), 
            "average_tcpWER": avg_tcpwer,
            "files": individual_results
        }

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved TCPWER results to {output_file}")
            
        return results