import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from query_faiss import FAISSQuery

from typing import List
import os

# OpenAI LLM
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_answer(question: str, passages: List[str]) -> str:
    """
    Generate an answer for a given question based on the provided passages.
    """
    content = "\n\n".join(passages)
    prompt = f"Use the following passages to answer the question strictly:\n{content}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2 # deterministic
    )

    return response.choices[0].message.content.strip()