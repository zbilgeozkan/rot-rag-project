import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from query_faiss import FAISSQuery
from llm_wrapper import generate_answer

if __name__ == "__main__":
    question = "What are the safety precautions for using this device?"

    # Retrieve passages from FAISS
    faiss_query = FAISSQuery()
    results = faiss_query.query(question, top_k=5)  # decrease top_k for fewer passages
    passages = [r["text"] for r in results]

    # Generate detailed answer using the LLM
    answer = generate_answer(question, passages)

    print("\nQuestion:", question)
    print("\nAnswer:", answer)

