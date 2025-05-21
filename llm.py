import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq  # Updated import
import os

# --- Configuration ---
INDEX_FILE = "./sop_index/faiss.index"
CHUNKS_FILE = "./sop_index/chunks.json"
META_FILE = "./sop_index/chunk_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Set Groq API Key from environment variable
api_key = "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"
client = Groq(api_key=api_key)

# --- Load Data ---
def load_index_and_data():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, chunks, metadata

# --- Retrieve Top Chunks ---
def retrieve_top_chunks(query, embedder, index, chunks, metadata, top_k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    for i in I[0]:
        results.append({
            "text": chunks[i],
            "source": metadata[i]["source"],
            "chunk_index": metadata[i]["chunk_index"]
        })
    return results

# --- Groq Completion ---
def call_groq(query, context_chunks):
    context_text = "\n\n".join(
        f"[{c['source']} | Chunk {c['chunk_index']}]:\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a compliance assistant for hospital SOPs. Your job is to help find the most accurate "
        "SOP/policy that applies to the context of the case, brief the clause with it's code (i.e. 4.2.2), what it says (word-by-word) and state your reason in a sentence as well. If multiple policies apply to a case provided, then do brief them in order."
        "These will be used as a reference in reporting so accuracy matters. Policy Code, Title is compulsory. NOTHING EXTRA NEEDED!"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Change this if using a different model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content

# --- Main Loop ---
if __name__ == "__main__":
    embedder = SentenceTransformer(EMBED_MODEL)
    index, chunks, metadata = load_index_and_data()

    while True:
        query = input("Ask a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        top_chunks = retrieve_top_chunks(query, embedder, index, chunks, metadata, top_k=TOP_K)
        print(top_chunks)
        response = call_groq(query, top_chunks)
        print("\n📘 Answer:\n", response, "\n")
        if chunks == "policy":
            chunks = print(chunks)

