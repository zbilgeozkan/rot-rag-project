# RoT RAG Project

This project demonstrates a simple Retrieval Augmented Generation (RAG) pipeline 
using **FAISS** for vector search and **SentenceTransformers** for embeddings.


## Project Structure
```
ROT-RAG-PROJECT/
├── data/                    # Documents and generated artifacts
│ ├── chunks.json            # Generated text chunks
│ ├── faiss_index.bin        # FAISS index file
│ ├── sample.pdf             # Example PDF document
│ └── test.txt               # Example TXT document
│
├── src/                     # Source code
│ ├── embed_faiss.py         # Build FAISS index from chunks
│ ├── ingest.py              # Process documents into chunks
│ └── query_faiss.py         # Query the FAISS index
│
├── tests/                   # Unit / integration tests
│   └── performance/
│       ├── test_faiss_speed.py   # Performance testing for FAISS
│       └── test_chunks.json      # Test chunks for benchmarking
│
├── .gitignore
├── Dockerfile               # Containerization support
├── LICENSE
├── pyproject.toml
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/zbilgeozkan/rot-rag-project.git
cd rot-rag-project
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Place your `.pdf` or `.txt` files inside the `data/` directory.

## Usage

### Step 1: Ingest documents
```bash
py src/ingest.py
```

Splits documents into chunks and saves them in: `data/chunks.json`.

---

### Step 2: Build FAISS index
```bash
py src/embed_faiss.py
```

Creates the FAISS index: `data/faiss_index.bin`.

---

### Step 3: Query the index
```bash
py src/query_faiss.py
```

You can update the query string inside `query_faiss.py`:

```python
results = faiss_query.query("Your question here")
```

## Example output
```json
{
  "text": "Example content from document...",
  "source": "sample.pdf",
  "page": 2,
  "distance": 0.12345
}
```

## Notes
- Replace `sample.pdf` and `test.txt` with your own content for meaningful results.

- Default embedding model: `all-MiniLM-L6-v2`.

- For large datasets, consider GPU FAISS (`faiss-gpu`).

- You can also run this project in a container using the provided Dockerfile.