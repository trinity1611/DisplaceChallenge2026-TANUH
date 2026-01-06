import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple
from utils.logger import LoggerSetup
from utils.helpers import FileIO, ConfigLoader
from utils.dataloader import SummarizationDataLoader
from src.Summarization import get_summarizer
from src.Summarization.batch_processor import SummarizationBatchProcessor
from src.metrics.metric_summarization import SummarizationMetrics

logger = LoggerSetup.get_logger("summarization")


def run_summarization(config_dir: str = "./config"):
    try:
        logger.info("="*80)
        logger.info("Starting Summarization Pipeline")
        logger.info("="*80)
        print("\n" + "="*80)
        print("STARTING SUMMARIZATION PIPELINE")
        print("="*80 + "\n")
        
        config = ConfigLoader.load_summarization_config(config_dir)
        
        csv_file = config.get("gt_path")
        if not csv_file:
            raise ValueError("CSV file path not provided. Please set 'gt_path' in summarization.yaml")
        
        logger.info(f"Using CSV file: {csv_file}")
        logger.info("Loaded configuration file")
        print(f"[STAGE] Using CSV file: {csv_file}")
        print(f"[STAGE] Loaded configuration file\n")
        
        data_loader = SummarizationDataLoader(config)
        records = data_loader.extract_and_load_asr(csv_file)
        logger.info(f"Extracted {len(records)} records from CSV with ASR files")
        print(f"[STAGE] Extracted {len(records)} records from CSV with ASR files\n")
        
        file_pairs = [(asr_text, rec_id) for rec_id, asr_text, _ in records]
        ground_truth_map = {rec_id: gt_summary for rec_id, _, gt_summary in records}
        
        summarization_output_dir = config.get("output", {}).get("summarization_dir", "./outputs/Summarization")
        metrics_output_dir = config.get("output", {}).get("metrics_dir", "./outputs/metrics")
        
        summarizer = get_summarizer(config)
        logger.info(f"Initialized summarization model: {config['model']['name']}")
        print(f"[STAGE] Initialized summarization model: {config['model']['name']}\n")
        
        summarization_processor = SummarizationBatchProcessor(
            summarizer,
            config,
            summarization_output_dir
        )
        logger.info("Processing summarization batch")
        print("[STAGE] Processing summarization batch\n")
        summarization_results = summarization_processor.process_batch(file_pairs)
        
        print("\n[STAGE] Calculating metrics for each file\n")
        metrics_results = []
        for summary_result in summarization_results:
            if summary_result["status"] == "success":
                rec_id = summary_result["rec_id"]
                generated_summary = FileIO.read_text_file(summary_result["output_file"])
                ground_truth = ground_truth_map.get(rec_id, "")
                
                if ground_truth:
                    file_metrics = SummarizationMetrics.calculate_file_metrics(
                        ground_truth, 
                        generated_summary
                    )
                    file_metrics["rec_id"] = rec_id
                    metrics_results.append(file_metrics)
                    logger.info(f"Calculated metrics for {rec_id}")
                    print(f"[STAGE] Calculated metrics for {rec_id}")
        
        print("\n[STAGE] Calculating batch metrics\n")
        batch_metrics = SummarizationMetrics.calculate_batch_metrics(metrics_results)
        logger.info(f"Calculated batch metrics for {len(metrics_results)} files")
        
        metrics_output_path = Path(metrics_output_dir)
        
        final_metrics = {
            "batch_metrics": batch_metrics,
            "file_metrics": metrics_results
        }
        
        SummarizationMetrics.save_metrics(
            final_metrics,
            metrics_output_path,
            "summarization_metrics.json"
        )
        logger.info("Saved consolidated metrics to summarization_metrics.json")
        print("[STAGE] Saved consolidated metrics to summarization_metrics.json\n")
        
        SummarizationMetrics.print_batch_summary(batch_metrics)
        
        logger.info("="*80)
        logger.info("Summarization Pipeline Completed Successfully")
        logger.info("="*80)
        print("="*80)
        print("SUMMARIZATION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return {
            "status": "success",
            "summarization_results": summarization_results,
            "metrics": batch_metrics
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"[ERROR] Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_summarization()

