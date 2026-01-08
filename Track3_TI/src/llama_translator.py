from typing import Dict, List, Any
import torch
from transformers import pipeline
from utils.logger import LoggerSetup

logger = LoggerSetup.get_logger("translator")

class LlamaTranslator:
    """
    LLaMA-based translator using Hugging Face text-generation pipeline.
    Prompt is kept EXACTLY as provided by the user.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model"]["name"]

        logger.info(f"Loading translation model: {self.model_name}")

        try:
            torch_dtype = torch.bfloat16
            if isinstance(config["model"].get("torch_dtype"), str):
                if config["model"]["torch_dtype"] == "float16":
                    torch_dtype = torch.float16
                elif config["model"]["torch_dtype"] == "bfloat16":
                    torch_dtype = torch.bfloat16

            
            self.pipe = pipeline(
                task="text-generation",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device_map=config["model"].get("device_map", "auto"),
                token=config["model"].get("hf_token"),
            )

            self.max_new_tokens = config.get("inference", {}).get(
                "max_new_tokens", 512
            )

            logger.info("Translation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            raise

    
    def translate(
        self,
        text: str,
        source_lang: str = "hi",
        target_lang: str = "en",
    ) -> str:
        """
        Translate a single text using the ORIGINAL user-provided prompt.
        """

        if text is None or str(text).strip() == "":
            return ""

        try:
            prompt = f"""
You are a professional Hindi → English medical translator.

TASK:
Translate the given Hindi medical conversation into clear, natural English.

STRICT RULES (follow strictly):
- Translate ALL Hindi and Hinglish into English.
- DO NOT copy Hindi words or sentences.
- DO NOT transliterate Hindi into English letters.
- DO NOT explain, summarize, analyze, or add information.
- DO NOT repeat sentences.
- Fix punctuation and sentence boundaries.
- Preserve medical meaning exactly.

OUTPUT FORMAT (MANDATORY):

<TRANSLATION>
English translation only.
</TRANSLATION>

Text:
{text}
""".strip()


            messages = [
                {"role": "system", "content": "You are a professional medical translator."},
                {"role": "user", "content": prompt},
            ]

            
            outputs = self.pipe(
                messages,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                num_beams=1,
                repetition_penalty=1.05,
                use_cache=True,
)


            
            response = outputs[0]["generated_text"][-1]["content"].strip()

            
            if "<TRANSLATION>" in response:
                translated = response.split("<TRANSLATION>")[-1]
            else:
                translated = response

            if "</TRANSLATION>" in translated:
                translated = translated.split("</TRANSLATION>")[0]

            return translated.strip()

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

   
    def batch_translate(
        self,
        texts: List[str],
        source_lang: str = "hi",
        target_lang: str = "en",
    ) -> List[str]:
        """
        Translate a list of texts sequentially.
        """

        results: List[str] = []

        for text in texts:
            try:
                translated = self.translate(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                results.append(translated)
            except Exception as e:
                logger.error(f"Failed to translate text: {str(e)}")
                results.append("")

        return results
