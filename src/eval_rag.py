import json
from query_faiss import FAISSQuery

def evaluate(test_file="data/test_cases.json", k=3):
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

        # Extract only the text fields
        combined_text = " ".join([r["text"] for r in results]).lower()

        # Check if expected keyword is in the results
        if expected.lower() in combined_text:
            correct += 1
        else:
            failed_cases.append({
                "question": question,
                "expected": expected,
                "got": [r["text"] for r in results]
            })

    # Calculate accuracy
    accuracy = correct / total * 100

    print(f"Total tests: {total}")
    print(f"Correct matches: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    if failed_cases:
        print("\nFailed test cases:")
        for fail in failed_cases:
            print(f"- Question: {fail['question']}")
            print(f"  Expected keyword: {fail['expected']}")
            print(f"  Retrieved results: {fail['got']}\n")

if __name__ == "__main__":
    evaluate(k=5)  # You can adjust k for more or fewer results