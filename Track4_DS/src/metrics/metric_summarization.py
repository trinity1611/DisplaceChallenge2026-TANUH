import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class SummarizationMetrics:
    
    @staticmethod
    def calculate_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            return {"error": "rouge_score not installed"}

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)

        return {
            "rougeL": scores['rougeL'].fmeasure
        }

    @staticmethod
    def calculate_bertscore(reference: str, hypothesis: str) -> Dict[str, float]:
        try:
            from bert_score import score
        except ImportError:
            return {"error": "bert_score not installed"}

        try:
            import warnings
            from transformers import logging
            
            logging.set_verbosity_error()
            warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
            
            precision, recall, f1 = score([hypothesis], [reference], lang="en", verbose=False)
            return {
                "bertscore_precision": precision.item(),
                "bertscore_recall": recall.item(),
                "bertscore_f1": f1.item()
            }
        except Exception as e:
            return {"bertscore_error": str(e)}

    @staticmethod
    def calculate_file_metrics(ground_truth: str, generated: str) -> Dict[str, Any]:
        metrics = {}
        
        try:
            metrics.update(SummarizationMetrics.calculate_rouge_l(ground_truth, generated))
        except Exception as e:
            metrics["rouge_error"] = str(e)

        try:
            metrics.update(SummarizationMetrics.calculate_bertscore(ground_truth, generated))
        except Exception as e:
            metrics["bertscore_error"] = str(e)

        metrics["ground_truth_length"] = len(ground_truth.split())
        metrics["generated_length"] = len(generated.split())
        metrics["length_ratio"] = len(generated.split()) / len(ground_truth.split()) if ground_truth else 0

        return metrics

    @staticmethod
    def calculate_batch_metrics(file_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not file_metrics:
            return {}

        batch_metrics = {
            "total_files": len(file_metrics),
            "timestamp": datetime.now().isoformat()
        }

        rougeL_scores = [m.get("rougeL", 0) for m in file_metrics if "rougeL" in m]
        bertscore_f1_scores = [m.get("bertscore_f1", 0) for m in file_metrics if "bertscore_f1" in m]

        if rougeL_scores:
            batch_metrics["avg_rougeL"] = sum(rougeL_scores) / len(rougeL_scores)
            batch_metrics["max_rougeL"] = max(rougeL_scores)
            batch_metrics["min_rougeL"] = min(rougeL_scores)

        if bertscore_f1_scores:
            batch_metrics["avg_bertscore_f1"] = sum(bertscore_f1_scores) / len(bertscore_f1_scores)
            batch_metrics["max_bertscore_f1"] = max(bertscore_f1_scores)
            batch_metrics["min_bertscore_f1"] = min(bertscore_f1_scores)

        avg_gen_length = sum(m.get("generated_length", 0) for m in file_metrics) / len(file_metrics)
        avg_ref_length = sum(m.get("ground_truth_length", 0) for m in file_metrics) / len(file_metrics)
        batch_metrics["avg_generated_length"] = avg_gen_length
        batch_metrics["avg_reference_length"] = avg_ref_length

        return batch_metrics

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], output_path: Path, filename: str) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    @staticmethod
    def print_batch_summary(batch_metrics: Dict[str, Any]) -> None:
        print("\n" + "="*80)
        print("BATCH METRICS SUMMARY")
        print("="*80)
        print(f"Total Files Processed: {batch_metrics.get('total_files', 0)}")
        print(f"Timestamp: {batch_metrics.get('timestamp', 'N/A')}")
        print("\nROUGE-L Scores:")
        print(f"  Average: {batch_metrics.get('avg_rougeL', 0):.4f}")
        print(f"  Maximum: {batch_metrics.get('max_rougeL', 0):.4f}")
        print(f"  Minimum: {batch_metrics.get('min_rougeL', 0):.4f}")
        print("\nBERTScore F1:")
        print(f"  Average: {batch_metrics.get('avg_bertscore_f1', 0):.4f}")
        print(f"  Maximum: {batch_metrics.get('max_bertscore_f1', 0):.4f}")
        print(f"  Minimum: {batch_metrics.get('min_bertscore_f1', 0):.4f}")
        print("\nLength Statistics:")
        print(f"  Avg Generated Length: {batch_metrics.get('avg_generated_length', 0):.2f} words")
        print(f"  Avg Reference Length: {batch_metrics.get('avg_reference_length', 0):.2f} words")
        print("="*80 + "\n")
