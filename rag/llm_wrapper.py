from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Device selection: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device set to use", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

def generate_answer(question: str, passages: List[str], max_new_tokens: int = 500) -> str:
    """
    Generate a clear, step-by-step answer for a question using the provided passages.
    """
    context = "\n\n".join(passages)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (step-by-step, with tips and precautions):"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()