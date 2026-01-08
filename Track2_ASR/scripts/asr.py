import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import LoggerSetup
from utils.helpers import ConfigLoader
from src.metrics.metric_asr import TranscriptionMetrics
from src.asr.transcribe_segments import run_transcription
from utils.logger import LoggerSetup

logger = LoggerSetup.get_logger("asr_pipeline")


def run_asr_pipeline(config_dir: str = "./config"):
    logger.info("="*80)
    logger.info("Starting ASR Pipeline")
    logger.info("="*80)

    asr_config = ConfigLoader.load_asr_config(config_dir)

    audio_folder = Path(asr_config["input_asr"]["input_audio_folder"])
    gt_folder = Path(asr_config["input_asr"]["input_gt"])
    rttm_folder = Path(asr_config["paths"]["diarization_output_dir"])
    seg_out_folder = Path(asr_config["output_asr"]["seg_out_folder"])
    fa_out_folder = Path(asr_config["output_asr"]["fa_out_folder"])
    metrics_output_dir = Path(asr_config["paths"]["metrics_output_dir"])
    wer_output_file = metrics_output_dir / "asr_cer_wer_results.json"
    cpwer_output_file = metrics_output_dir / "asr_tcpwer_per_file.json"
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'audio_folder {audio_folder}')
    logger.info(f'rttm_folder{rttm_folder}')
    logger.info(f'gt_folder {gt_folder}')

    logger.info("Starting RTTM-based ASR transcription")
    run_transcription(
        audio_folder=audio_folder,
        rttm_folder=rttm_folder,
        seg_out_folder=seg_out_folder,
        fa_out_folder=fa_out_folder,
        model_config=asr_config
    )
    logger.info("Transcription completed successfully")
    
    # ----------------------------------------------------
    # Calculate metrics if ground-truth exists
    # ----------------------------------------------------
    ground_truth_exists = gt_folder.exists() and any(gt_folder.rglob("*.txt"))

    if ground_truth_exists:
        logger.info("Calculating WER / CER")
        metrics = TranscriptionMetrics.calculate_wer_cer(asr_config, output_file=str(wer_output_file))
        logger.info(
                    f"ASR Metrics | Records Processed: {metrics['records processed']} "
                    f"Average CER: {metrics['average_cer']:.3f} "
                    f"Average WER: {metrics['average_wer']:.3f}"
                    )

        print(
            f"ASR Metrics | Records Processed: {metrics['records processed']} "
            f"Average CER: {metrics['average_cer']:.3f} "
            f"Average WER: {metrics['average_wer']:.3f}"
            )

        logger.info("Calculating tcpWER")
        metrics = TranscriptionMetrics.compute_tcpwer(asr_config, output_file=str(cpwer_output_file), eval_speaker=False)
        logger.info(
                f"SD + ASR Metrics | Records Processed: {metrics['records processed']} "
                f"Average tcpWER: {metrics['average_tcpWER']:.3f}"
                )

        print(
            f"SD + ASR Metrics | Records Processed: {metrics['records processed']} "
            f"Average tcpWER: {metrics['average_tcpWER']:.3f}"
            )
    else:
        logger.info(f"No ground truth found at {gt_folder}")
        logger.info(f"Skipping WER / CER / tcpWER computation")
        print(f"No ground truth found at {gt_folder}")
        print("Skipping WER / CER / tcpWER computation")
        metrics = None

    logger.info("="*80)
    logger.info("ASR Pipeline Completed Successfully")
    logger.info("="*80)
    return {"status": "success", "metrics": metrics}    


if __name__ == "__main__":
    run_asr_pipeline()
