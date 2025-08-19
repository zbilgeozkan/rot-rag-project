import json
import faiss
from sentence_transformers import SentenceTransformer

# Load chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load FAISS index
index = faiss.read_index("data/faiss_index.bin")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed query
query = "Your query text here"
query_vec = model.encode([query], convert_to_numpy=True)

# Search in FAISS index
k = 3  # number of nearest neighbors
distances, indices = index.search(query_vec, k)

# Display results
for i, idx in enumerate(indices[0]):
    print(f"### Similar chunk {i + 1} ###")
    print(f"Source: {chunks[idx]['source']}, Page: {chunks[idx]['page']}")
    print(chunks[idx]['text'])
    print()