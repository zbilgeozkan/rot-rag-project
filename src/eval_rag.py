import sys
import os
from pathlib import Path
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from query_faiss import FAISSQuery

def evaluate(test_file="data/test_cases.json", k=3, output_failed="data/failed_cases.json"):
    # Load test cases
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    faiss_query = FAISSQuery()
    total = len(test_cases)
    correct = 0
    failed_cases = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected_keyword"]

        # Query the FAISS index
        results = faiss_query.query(question, top_k=k)

        # Check in text, title, and source
        found = False
        for r in results:
            combined_fields = " ".join([r.get("text", ""), r.get("title", ""), r.get("source", "")]).lower()
            if expected.lower() in combined_fields:
                found = True
                break

        if found:
            correct += 1
        else:
            failed_cases.append({
                "question": question,
                "expected": expected,
                "results": results
            })

    # Calculate accuracy
    accuracy = correct / total * 100

    print(f"Total tests: {total}")
    print(f"Correct matches: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save failed cases to JSON
    Path(output_failed).parent.mkdir(parents=True, exist_ok=True)
    with open(output_failed, "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)
    
    if failed_cases:
        print(f"\nFailed test cases saved to {output_failed}")

if __name__ == "__main__":
    evaluate(k=5)  # You can adjust k for more or fewer results