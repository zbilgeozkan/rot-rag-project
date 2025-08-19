import json
from sentence_transformers import SentenceTransformer
import faiss

# Load chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2') # fast and small model

# Embed texts
texts = [c['text'] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)

print(f"Index built with {index.ntotal} vectors.")

# Save the index
faiss.write_index(index, "data/faiss_index.bin")
print("Index saved to data/faiss_index.bin")