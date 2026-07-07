"""
DISPLACE MedAI – Topic Extraction Service (Track 3)
=====================================================
Wraps the Qwen 2.5-3B-Instruct model to extract medical topics
from conversation transcripts.
"""

import logging
import time
import re
from typing import Dict, Any, Optional

import torch

from backend.app.config import settings

logger = logging.getLogger("displace.topic_extraction")


class TopicExtractionService:
    """Medical topic extraction using Qwen (Track 3)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the Qwen model and tokenizer."""
        if self._loaded:
            logger.info("Topic extraction model already loaded, skipping.")
            return

        model_id = settings.qwen_model_id
        hf_token = settings.hf_token or None

        logger.info(f"Loading topic extraction model: {model_id}")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token,
        )
        self._model.eval()
        self._loaded = True
        logger.info("Topic extraction model loaded successfully")

    def unload_model(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Topic extraction model unloaded")

    def _build_prompt(self, text: str) -> str:
        """Build the medical topic extraction prompt (from Track3)."""
        return f"""You are a medical classifier.

Extract ONLY the patient's ongoing health problems.
Exclude family history, past illnesses that has been solved, and explanations.

Return ONLY a comma-separated list of health problem topics in short medical relevant words for each conversation text.

Conversation:
{text}

Problems:""".strip()

    def run(self, transcript: str) -> Dict[str, Any]:
        """
        Extract medical topics from a transcript.

        Args:
            transcript: Full conversation transcript text

        Returns:
            {
                "topics": str,       # Comma-separated topics
                "topics_list": [str], # List of individual topics
                "elapsed_s": float,
            }
        """
        start_time_wall = time.time()

        if not self._loaded:
            self.load_model()

        logger.info("Extracting medical topics from transcript")

        prompt = self._build_prompt(transcript)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=False,
            )

        # Decode only the generated tokens (skip prompt)
        prediction = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # Clean up prediction (from Track3's predictor.py logic)
        prediction = prediction.lower().strip()
        prediction = prediction.split("\n")[0]
        prediction = re.sub(r"[^a-z0-9, ]+", "", prediction)
        topics_list = [p.strip() for p in prediction.split(",") if p.strip()]
        topics_str = ", ".join(topics_list)

        elapsed = time.time() - start_time_wall

        logger.info(f"Topics extracted: {topics_str} ({elapsed:.1f}s)")

        return {
            "topics": topics_str,
            "topics_list": topics_list,
            "elapsed_s": elapsed,
        }


# Singleton instance
topic_extraction_service = TopicExtractionService()
