from typing import Dict, List, Any
from pathlib import Path
from utils.logger import LoggerSetup
from utils.helpers import FileIO

logger = LoggerSetup.get_logger("summarization")


class SummarizationBatchProcessor:
    def __init__(self, summarizer, config: Dict[str, Any], output_dir: str):
        self.summarizer = summarizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_pattern = config.get("output", {}).get("summary_pattern", "{rec_id}_summary.txt")

    @staticmethod
    def clean_text(text: str) -> str:
        text = ' '.join(text.split())
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()

    def process_file(self, input_file: str, rec_id: str) -> Dict[str, Any]:
        try:
            text = FileIO.read_text_file(input_file)
            text = self.clean_text(text)
            
            logger.info(f"Summarizing {rec_id}")
            print(f"[STAGE] Summarizing {rec_id}")
            summary = self.summarizer.summarize(text)
            
            output_filename = self.summary_pattern.format(rec_id=rec_id)
            output_file = self.output_dir / output_filename
            FileIO.write_text_file(str(output_file), summary)
            
            logger.info(f"Summarization completed for {rec_id}")
            print(f"[STAGE] Summarization completed for {rec_id}")
            
            return {
                "rec_id": rec_id,
                "input_file": input_file,
                "output_file": str(output_file),
                "status": "success",
                "input_length": len(text.split()),
                "output_length": len(summary.split())
            }
        except Exception as e:
            logger.error(f"Summarization failed for {rec_id}: {str(e)}")
            print(f"[ERROR] Summarization failed for {rec_id}: {str(e)}")
            return {
                "rec_id": rec_id,
                "status": "error",
                "error": str(e)
            }

    def process_batch(self, file_pairs: List[tuple]) -> List[Dict[str, Any]]:
        results = []
        for input_file, rec_id in file_pairs:
            result = self.process_file(input_file, rec_id)
            results.append(result)
        return results
