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
        "text": c["text"],
        "source": c["source"],
        "page": c["page"],
        "title": c.get("title", "Unknown")
    }
    for c in chunks
]

with open("data/faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Metadata saved to data/faiss_metadata.json")

# Retrieval Function for Evaluation
def load_faiss_index(index_path="data/faiss_index.bin", metadata_path="data/faiss_metadata.json"):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve_top_k(query, k=3, index_path="data/faiss_index.bin", metadata_path="data/faiss_metadata.json"):
    """Retrieve top-k most similar passages for a given query, with scores."""
    index, metadata = load_faiss_index(index_path, metadata_path)

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata):
            results.append({
                "text": metadata[idx]["text"],
                "score": float(score)  # FAISS distance (smaller the better)
            })
    return results
