import json
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from embed_faiss import retrieve_top_k  # my retrieval function (e.g.: query → top k chunks)

K = 3  # how many top-k passages to return

def load_gold_passages(path="tests/evaluation/gold_passages.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hit_at_k(retrieved, gold_passages):
    """Check if any gold passage is in retrieved results."""
    for gold in gold_passages:
        if any(gold.lower() in r.lower() for r in retrieved):
            return 1
    return 0

@pytest.mark.parametrize("sample", load_gold_passages())
def test_retrieval_quality(sample):
    query = sample["query"]
    gold_passages = sample["gold_passages"]

    retrieved = retrieve_top_k(query, k=K)  # my retrieval function should return chunks (text)
    retrieved_texts = [r for r in retrieved]

    # Evaluate hit@K
    hit = hit_at_k(retrieved_texts, gold_passages)
    assert hit == 1, f"Query failed: {query}\nExpected one of {gold_passages}\nGot {retrieved_texts}"
