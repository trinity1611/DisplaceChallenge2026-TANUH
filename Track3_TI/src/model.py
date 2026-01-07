import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer



def load_qwen(base_model: str, adapter_model: str):
    """
    Load Qwen base model with LoRA adapter for topic extraction.
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, adapter_model)

    model.eval()
    return tokenizer, model


def load_embedder(model_name: str):
    """
    Load sentence embedding model (CPU-friendly).
    """
    return SentenceTransformer(model_name)
