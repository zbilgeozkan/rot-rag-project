from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Device selection: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device set to use", device)

# Load tokenizer and model
model_name = "stabilityai/stablelm-zephyr-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)

def generate_answer(question: str, passages: List[str], max_new_tokens: int = 500) -> str:
    """
    Generate a clear, step-by-step answer for a question using the provided passages.
    """
    context = "\n\n".join(passages)

    # Prompt format - more natural for chat-tuned models
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
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
        temperature=0.7,
        top_p=0.9,
        do_sample=True,   # Zephyr için sampling genelde daha iyi
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()
