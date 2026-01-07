import torch
import re


def build_prompt(text: str) -> str:
    return f"""
You are a medical classifier.

Extract ONLY the patient's ongoing health problems.
Exclude family history, past illnesses that has been solved , and explanations.

Return ONLY a comma-separated list of health problem topics in short medical relevant words for each conversation text.

Conversation:
{text}

Problems:
""".strip()


@torch.no_grad()
def predict_topic(
    text: str,
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float
) -> str:

    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False
    )

    prediction = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )


    prediction = prediction.lower().strip()
    prediction = prediction.split("\n")[0]
    prediction = re.sub(r"[^a-z0-9, ]+", "", prediction)

    prediction = ",".join(
        [p.strip() for p in prediction.split(",") if p.strip()]
    )

    return prediction
