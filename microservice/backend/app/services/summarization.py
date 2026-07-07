"""
DISPLACE MedAI – Dialogue Summarization Service (Track 4)
============================================================
Wraps the LLaMA 3.2-3B-Instruct model to generate concise
medical dialogue summaries from conversation transcripts.
"""

import logging
import time
from typing import Dict, Any, Optional

import torch

from backend.app.config import settings

logger = logging.getLogger("displace.summarization")


class SummarizationService:
    """Medical dialogue summarization using LLaMA (Track 4)."""

    def __init__(self):
        self._pipe = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the LLaMA summarization pipeline."""
        if self._loaded:
            logger.info("Summarization model already loaded, skipping.")
            return

        model_id = settings.llama_model_id
        hf_token = settings.hf_token or None

        logger.info(f"Loading summarization model: {model_id}")

        from transformers import pipeline as hf_pipeline
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self._pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
            device_map="auto",
            token=hf_token,
        )

        self._loaded = True
        logger.info("Summarization model loaded successfully")

    def unload_model(self) -> None:
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._loaded = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Summarization model unloaded")

    def _build_prompt(self, text: str) -> str:
        """Build the summarization prompt (from Track4's summarizer.py)."""
        return f"""{text}

Task: Summarize the above conversation in English as a concise, factual patient summary and follow the format with these two tags <SUMMARY> and </SUMMARY>, the summary should be between these two tags and there should not be anything extra.

STRICT INSTRUCTIONS:
1. Use **third-person narration** only (e.g., "The patient reports...", "The doctor advises...").
2. Include **ONLY information explicitly mentioned** in the conversation.
3. Structure the summary to cover: **Chief Complaint**, **History of Present Illness**, **Diagnosis** (if any), and **Treatment Plan**.
4. Include **objective measurements** (e.g., BP, temperature) and **medications** if stated.
5. Do **NOT** hallucinate or add outside information.
6. The output must be **extremely concise** (approx. 100-130 words) and **clinically accurate**."""

    def run(self, transcript: str) -> Dict[str, Any]:
        """
        Generate a medical dialogue summary.

        Args:
            transcript: Full conversation transcript text

        Returns:
            {
                "summary": str,
                "elapsed_s": float,
            }
        """
        start_time_wall = time.time()

        if not self._loaded:
            self.load_model()

        logger.info("Generating dialogue summary")

        prompt = self._build_prompt(transcript)

        messages = [
            {"role": "system", "content": "You are a Summary Generator."},
            {"role": "user", "content": prompt},
        ]

        generation_args = {
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "pad_token_id": self._pipe.tokenizer.eos_token_id,
            "eos_token_id": self._pipe.tokenizer.eos_token_id,
        }

        outputs = self._pipe(messages, **generation_args)
        response = outputs[0]["generated_text"][-1]["content"].strip()

        # Extract summary between tags (from Track4's logic)
        summary = response
        if "<SUMMARY>" in summary:
            summary = summary.split("<SUMMARY>")[-1]
        if "</SUMMARY>" in summary:
            summary = summary.split("</SUMMARY>")[0]
        summary = summary.strip()

        elapsed = time.time() - start_time_wall

        logger.info(f"Summary generated ({len(summary)} chars, {elapsed:.1f}s)")

        return {
            "summary": summary,
            "elapsed_s": elapsed,
        }


# Singleton instance
summarization_service = SummarizationService()
