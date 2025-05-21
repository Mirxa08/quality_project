import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Configuration ---
TEXT_DIR = "Clean"
INDEX_FILE = "./sop_index/faiss.index"
CHUNKS_FILE = "./sop_index/chunks.json"
META_FILE = "./sop_index/chunk_metadata.json"

CHUNK_SIZE = 300  # words
OVERLAP = 50      # words
EMBED_MODEL = "all-MiniLM-L6-v2"

os.makedirs("./sop_index", exist_ok=True)

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def process_documents(text_dir):
    all_chunks = []
    metadata = []
    for fname in os.listdir(text_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
                full_text = f.read()
            chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "source": fname,
                    "chunk_index": i
                })
    return all_chunks, metadata

def build_faiss_index(chunks, embedder):
    print("🔍 Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

if __name__ == "__main__":
    print("📚 Processing cleaned SOPs...")
    chunks, metadata = process_documents(TEXT_DIR)

    print(f"📦 Total chunks: {len(chunks)}")

    embedder = SentenceTransformer(EMBED_MODEL)
    index, embeddings = build_faiss_index(chunks, embedder)

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ FAISS index saved to: {INDEX_FILE}")

    # Save chunks and metadata
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("✅ Chunk text and metadata saved.")
