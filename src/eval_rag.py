import json
from query_faiss import FAISSQuery

def evaluate(test_cases, top_k=3):
    faiss_query = FAISSQuery()
    correct = 0

    for case in test_cases:
        question = case["question"]
        expected = case["expected_keyword"]

        print(f"Q: {question}")
        print(f"Expected keyword: {expected}")

        results = faiss_query.query(question, top_k=top_k)

        found = False
        for r in results:
            print(f" - {r['text'][:100]}...")
            if expected.lower() in r["text"].lower():
                found = True

        if found:
            correct += 1
            print("True!")
        else:
            print("False!")

        print("-" * 50)

    accuracy = correct / len(test_cases) * 100
    print(f"Overall accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # Read scenarios from JSON
    with open("data/test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    evaluate(test_cases)