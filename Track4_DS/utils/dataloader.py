from typing import List, Tuple
from pathlib import Path
import pandas as pd
from utils.logger import LoggerSetup

class SummarizationDataLoader:
    def __init__(self, config: dict):
        self.logger = LoggerSetup.get_logger("summarization")
        self.config = config
        self.rec_id_column = config.get("columns", {}).get("rec_id", "rec_id")
        self.gt_summary_column = config.get("columns", {}).get("gt_summary", "gt_summary")
        self.asr_dir = Path(config.get("input_path", "./outputs/ASR/fullaudio"))
        self.asr_filename_pattern = config.get("output", {}).get("asr_pattern", "{rec_id}_fullaudio_transcription.txt")

    def load_csv(self, csv_file: str) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading CSV file: {csv_file}")
            print(f"[STAGE] Loading CSV file: {csv_file}")
            df = pd.read_csv(csv_file, encoding='utf-8')
            self.logger.info(f"Loaded {len(df)} records from CSV")
            print(f"[STAGE] Loaded {len(df)} records from CSV")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {str(e)}")
            print(f"[ERROR] Failed to load CSV: {str(e)}")
            raise

    def extract_and_load_asr(self, csv_file: str) -> List[Tuple[str, str, str]]:
        try:
            df = self.load_csv(csv_file)
            
            if self.rec_id_column not in df.columns:
                raise ValueError(f"Column '{self.rec_id_column}' not found in CSV")
            if self.gt_summary_column not in df.columns:
                raise ValueError(f"Column '{self.gt_summary_column}' not found in CSV")
            
            records = []
            skipped_count = 0
            
            for idx, row in df.iterrows():
                rec_id = str(row[self.rec_id_column]).strip()
                gt_summary = str(row[self.gt_summary_column]).strip()
                
                asr_filename = self.asr_filename_pattern.format(rec_id=rec_id)
                asr_file_path = self.asr_dir / asr_filename
                
                try:
                    if not asr_file_path.exists():
                        self.logger.warning(f"ASR file not found for {rec_id}: {asr_file_path}. Skipping.")
                        print(f"[WARNING] ASR file not found for {rec_id}: {asr_file_path}. Skipping.")
                        skipped_count += 1
                        continue
                    
                    records.append((rec_id, asr_file_path, gt_summary))
                    self.logger.debug(f"Loaded ASR for record: {rec_id}")
                except Exception as e:
                    self.logger.error(f"Error processing record {rec_id}: {str(e)}. Skipping.")
                    print(f"[ERROR] Error processing record {rec_id}: {str(e)}. Skipping.")
                    skipped_count += 1
                    continue
            
            self.logger.info(f"Successfully loaded {len(records)} records. Skipped {skipped_count} records.")
            print(f"[STAGE] Successfully loaded {len(records)} records. Skipped {skipped_count} records.")
            return records
        except Exception as e:
            self.logger.error(f"Failed to extract and load ASR: {str(e)}")
            print(f"[ERROR] Failed to extract and load ASR: {str(e)}")
            raise
