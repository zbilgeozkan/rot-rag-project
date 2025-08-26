import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSQuery:
    def __init__(self, index_path="data/faiss_index.bin", metadata_path="data/faiss_metadata.json", model_name='all-MiniLM-L6-v2'):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunks metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Initialize embedding model
        self.model = SentenceTransformer(model_name)

    def query(self, text, top_k=5):
        # Embed query text
        query_vec = self.model.encode([text], convert_to_numpy=True)

        # Search in FAISS index
        distances, indices = self.index.search(query_vec, top_k)

        # Return matched chunks with metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue  # IndexError Fix
            chunk_meta = self.metadata[idx]
            results.append({
                "text": chunk_meta.get("text", ""),
                "source": chunk_meta.get("source", ""),
                "page": chunk_meta.get("page", 0),
                "title": chunk_meta.get("title", "Unknown"),
                "distance": float(dist)
            })
        return results

# Test
if __name__ == "__main__":
    faiss_query = FAISSQuery()
    results = faiss_query.query("How do I wear the Gear VR headset?", top_k=5)
    
    # Save in JSON file
    output_path = "data/query_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")
