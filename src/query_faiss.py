import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class FAISSQuery:
    def __init__(self, index_path="data/faiss_index.bin", chunks_path="data/chunks.json", model_name='all-MiniLM-L6-v2'):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunks metadata
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Initialize embedding model
        self.model = SentenceTransformer(model_name)

    def query(self, text, top_k=5):
        # Embed query text
        query_vec = self.model.encode([text], convert_to_numpy=True)

        # Search in FAISS index
        distances, indices = self.index.search(query_vec, top_k)

        # Return matched chunks
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "distance": float(dist)
            })
        return results

# Test
if __name__ == "__main__":
    faiss_query = FAISSQuery()
    results = faiss_query.query("Araç fren sistemini nasıl kontrol etmeliyim?", top_k=5)
    
    # Save in JSON file
    output_path = "data/query_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}") 
