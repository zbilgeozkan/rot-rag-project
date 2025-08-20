import json
from query_faiss import FAISSQuery

# Test dataset: query and expected answer (gold answer)
test_cases = [
    {
        "query": "Aracın yakıt türü ne olmalı?",
        "expected": "yakıt"
    },
    {
        "query": "Lastik basıncını ne zaman kontrol etmeliyim?",
        "expected": "lastik"
    },
    {
        "query": "Bilgisayarımın performansını nasıl artırabilirim?",
        "expected": "bilgisayar"
    },
    {
        "query": "Yangın tüpü evde bulundurulmalı mı?",
        "expected": "yangın"
    },
]

def evaluate(top_k=3):
    faiss_query = FAISSQuery()
    correct = 0

    for case in test_cases:
        query = case["query"]
        expected = case["expected"]

        results = faiss_query.query(query, top_k=top_k)

        # Do any of the results contain the expected word
        found = any(expected.lower() in r["text"].lower() for r in results)

        print(f"Q: {query}")
        print(f"Expected keyword: {expected}")
        print(f"Results: ")
        for r in results:
            print(" -", r["text"][:100], "...")
        print("True!" if found else "False!")
        print("-" * 50)

        if found:
            correct += 1

    accuracy = correct / len(test_cases) * 100
    print(f"Overall accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()