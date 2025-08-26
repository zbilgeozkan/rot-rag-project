import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from embed_faiss import retrieve_top_k

def hit_at_k(retrieved, gold_passages):
    for gold in gold_passages:
        if any(gold.lower() in r["text"].lower() for r in retrieved):
            return 1
    return 0

def precision_at_k(retrieved, gold_passages):
    retrieved_texts = [r["text"].lower() for r in retrieved]
    gold_texts = [g.lower() for g in gold_passages]
    hits = sum(1 for r in retrieved_texts if any(g in r for g in gold_texts))
    return hits / len(retrieved) if retrieved else 0

def recall_at_k(retrieved, gold_passages):
    retrieved_texts = [r["text"].lower() for r in retrieved]
    gold_texts = [g.lower() for g in gold_passages]
    hits = sum(1 for g in gold_texts if any(g in r for r in retrieved_texts))
    return hits / len(gold_passages) if gold_passages else 0

def evaluate(gold_path="tests/evaluation/gold_passages.json", k=3):
    with open(gold_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total_hit, total_prec, total_rec = 0, 0, 0

    for sample in samples:
        query = sample["query"]
        gold = sample["gold_passages"]

        retrieved = retrieve_top_k(query, k=k)

        hit = hit_at_k(retrieved, gold)
        prec = precision_at_k(retrieved, gold)
        rec = recall_at_k(retrieved, gold)

        total_hit += hit
        total_prec += prec
        total_rec += rec

        print(f"\nQuery: {query}")
        print(f"Hit@{k}: {hit}, Precision@{k}: {prec:.2f}, Recall@{k}: {rec:.2f}")
        print(f"Top-{k} Retrieved: {[r['text'][:80] for r in retrieved]}")

    n = len(samples)
    print("\n--- Overall Evaluation ---")
    print(f"Hit@{k}: {total_hit/n:.2f}")
    print(f"Precision@{k}: {total_prec/n:.2f}")
    print(f"Recall@{k}: {total_rec/n:.2f}")

if __name__ == "__main__":
    evaluate(k=3)
