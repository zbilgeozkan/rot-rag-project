import json
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load chunks
with open("tests/performance/test_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast and small

# Embed and time it
start = time.time()
embeddings = model.encode(texts)
print("Embedding time:", time.time() - start, "seconds")

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

start = time.time()
index.add(np.array(embeddings, dtype='float32'))
print("FAISS index build time:", time.time() - start, "seconds")
