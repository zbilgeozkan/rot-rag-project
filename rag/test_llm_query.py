import sys
import os
import time
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__)))
from query_faiss import FAISSQuery
from llm_wrapper import generate_answer

if __name__ == "__main__":
    question = "How can I remove the mobile device from the headset?"

    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Retrieve passages from FAISS
    faiss_query = FAISSQuery()
    results = faiss_query.query(question, top_k=5)  # decrease top_k for fewer passages
    passages = [r["text"] for r in results]

    # Measure time
    start_time = time.time()

    # Generate detailed answer using the LLM
    answer = generate_answer(question, passages)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print to console
    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print(f"\nTime: {elapsed_time:.2f} seconds")

    # Save to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(data_dir, f"output_{timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n\n")
        f.write(f"Answer:\n{answer}\n\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n")

    print(f"\nOutput saved to: {output_file}")
