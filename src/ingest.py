import os
import json
from PyPDF2 import PdfReader
from collections import defaultdict

# Read TXT
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Read PDF
def read_pdf_file(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Read all files
def read_all_files(data_dir="data"):
    docs = {}
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.lower().endswith(".txt"):
            docs[filename] = read_txt_file(path)
        elif filename.lower().endswith(".pdf"):
            docs[filename] = read_pdf_file(path)
    return docs

# Chunk
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Metadata & pipeline
def ingest_all(data_dir="data"):
    all_meta_chunks = []
    docs = read_all_files(data_dir)
    for filename, content in docs.items():
        chunks = chunk_text(content)
        for i, c_text in enumerate(chunks):
            chunk = {
                "text": c_text,
                "source": filename,
                "page": i
            }
            all_meta_chunks.append(chunk)
    return all_meta_chunks

# Test + JSON save
if __name__ == "__main__":
    all_chunks = ingest_all()

    # Display first 5 chunks based on file in terminal
    chunks_by_file = defaultdict(list)
    for c in all_chunks:
        chunks_by_file[c['source']].append(c)

    for filename, chunks in chunks_by_file.items():
        print(f"\n### First 5 chunks for {filename} ###")
        for chunk in chunks[:5]:
            print(chunk)

    # Saving to JSON
    import json
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_chunks)} chunks to data/chunks.json")