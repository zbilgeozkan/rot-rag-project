import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Automatic device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract texts and metadata
texts = [c['text'] for c in chunks]

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device) # fast and small model

# Check if embeddings already exist
embeddings_path = "data/embeddings.npy"
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
    print("Embeddings loaded from cache.")
else:
    # Embed texts in batches
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64  # Batch size: can be increased depending on hardware
    )
    np.save(embeddings_path, embeddings)
    print("Embeddings computed and saved to cache.")

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)

print(f"Index built with {index.ntotal} vectors.")

# Save the index
faiss_index_path = "data/faiss_index.bin"
faiss.write_index(index, faiss_index_path)

print(f"Index saved to {faiss_index_path}")

# Save metadata
metadata = [
    {
        "id": c["id"],
        "source": c["source"],
        "page": c["page"],
        "title": c.get("title", "Unknown")
    }
    for c in chunks
]

with open("data/faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Metadata saved to data/faiss_metadata.json")