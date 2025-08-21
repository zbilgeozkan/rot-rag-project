import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from query_faiss import FAISSQuery
from llm_wrapper import generate_answer

if __name__ == "__main__":
    question = "How do I wear the Gear VR headset?"
    
    # Pull passages with FAISS
    faiss_query = FAISSQuery()
    results = faiss_query.query(question, top_k=5)
    passages = [r["text"] for r in results]
    
    # Generate answer using LLM
    answer = generate_answer(question, passages)
    
    print("\nQuestion:", question)
    print("\nAnswer:", answer)