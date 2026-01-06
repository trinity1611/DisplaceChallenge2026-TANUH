from typing import Dict, List, Any
from pathlib import Path
import torch
from transformers import pipeline
from utils.logger import LoggerSetup

logger = LoggerSetup.get_logger("summarizer")


class Summarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model"]["name"]
        
        import transformers
        import numpy as np
        import random
        seed = config.get("optimization", {}).get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        transformers.set_seed(seed)
        
        logger.info(f"Loading summarization model: {self.model_name}")
        
        try:
            torch_dtype = torch.bfloat16
            if isinstance(config["model"].get("torch_dtype"), str):
                if config["model"]["torch_dtype"] == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif config["model"]["torch_dtype"] == "float16":
                    torch_dtype = torch.float16
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device_map=config["model"].get("device_map", "auto"),
                token=config["model"].get("hf_token")
            )
            
            self.max_new_tokens = config["inference"].get("max_new_tokens", 512)
            
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {str(e)}")
            raise

    def summarize(self, text: str, language: str = "en") -> str:
        try:
            prompt = f"""{text}

            Task: Summarize the above conversation in English as a concise, factual patient summary and follow the format with these two tags <SUMMARY> and </SUMMARY>, the summary should be between these two tags and there should not be anything extra. 

            STRICT INSTRUCTIONS:
            1. Use **third-person narration** only (e.g., "The patient reports...", "The doctor advises...").
            2. Include **ONLY information explicitly mentioned** in the conversation.
            3. Structure the summary to cover: **Chief Complaint**, **History of Present Illness**, **Diagnosis** (if any), and **Treatment Plan**.
            4. Include **objective measurements** (e.g., BP, temperature) and **medications** if stated.
            5. Do **NOT** hallucinate or add outside information.
            6. The output must be **extremely concise** (approx. 100-130 words) and **clinically accurate**.
            """

            
            messages = [
                {"role": "system", "content": "You are a Summary Generator."},
                {"role": "user", "content": prompt}
            ]
            
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.config["inference"].get("do_sample", True),
                "top_k": self.config["inference"].get("top_k", 50),
                "repetition_penalty": self.config["inference"].get("repetition_penalty", 1.0),
                "pad_token_id": self.pipe.tokenizer.eos_token_id,
                "eos_token_id": self.pipe.tokenizer.eos_token_id,
            }
            
            outputs = self.pipe(
                messages,
                **generation_args
            )
            
            response = outputs[0]["generated_text"][-1]["content"].strip()
            
            summary = response
            if "<SUMMARY>" in summary:
                summary = summary.split("<SUMMARY>")[-1]
            
            if "</SUMMARY>" in summary:
                summary = summary.split("</SUMMARY>")[0]
            
            summary = summary.strip()
            
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            raise

    def batch_summarize(self, texts: List[str], language: str = "en") -> List[str]:
        results = []
        for text in texts:
            try:
                summary = self.summarize(text, language)
                results.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize text: {str(e)}")
                results.append("")
        return results
