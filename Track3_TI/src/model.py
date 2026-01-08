import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


def load_qwen(base_model: str):
   

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model


def load_embedder(model_name: str):
    
    return SentenceTransformer(model_name)
