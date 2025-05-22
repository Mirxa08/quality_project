import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

INDEX_FILE = "./sop_index/faiss.index"
CHUNKS_FILE = "./sop_index/chunks.json"
META_FILE = "./sop_index/chunk_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

api_key = st.secrets["api"]["groq_key"]
client = Groq(api_key=api_key)

def load_index_and_data():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, chunks, metadata

def retrieve_top_chunks(query, embedder, index, chunks, metadata, top_k=TOP_K):
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

def call_groq(history, query, context_chunks):
    context_text = "\n\n".join(
        f"[{c['source']} | Chunk {c['chunk_index']}]:\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a hospital SOP compliance assistant.\n"
        "- Your goal is to identify the most relevant SOP or policy clauses that applies to the given situation. can be more than one\n"
        "- Cite the SOP and clause clearly. STRICTLY no need to state the chunk\n"
        "- It is vital to state the clause (i.e. 4.2.1) as what it says whenever you mention it.\n"
        "- Write short, direct, and fact-based answers suitable for streamlit UI.\n"
        "- Do not add long explanations or unnecessary info. BUT you can if you need to define a whole procedure (can check next clauses as the process defeined in clauses is in a orderly manner\n"
        "- Format the answer cleanly for professional use.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"})

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Policy Encyclopedia", layout="wide")
st.title("🏥 Policy Encyclopedia")

# Show toast once per session
if "welcome_shown" not in st.session_state:
    st.toast("💡 Ask a case-related question to find relevant policies to that scenario!", icon="💡")
    st.session_state.welcome_shown = True

with st.expander("ℹ️ About this Assistant"):
    st.markdown("""
    **Welcome to the Policy Encyclopedia!**  
    - Enter a **case scenario or compliance question**.  
    - The system will search SOP documents and return **relevant clauses**.  
    - Each result will **cite clause numbers** and **document names** relevant to the case.
    - System can extract more than one result in some cases so be sure to give it a read.  
    - Click **🆕 New Case** to start over.

    ✅ Designed for hospital QA, audits, and investiagtional purposes.
    
    
    ***Note***: This is an early model not having all the policies included so results may vary.
    """)

# Init session state
if "history" not in st.session_state:
    st.session_state.history = []

if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer(EMBED_MODEL)

if "index_data" not in st.session_state:
    st.session_state.index_data = load_index_and_data()

# New Case button
if st.button("🆕 New Case"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter your SOP-related question...")

if user_input:

    st.chat_message("user").markdown(user_input)

    index, chunks, metadata = st.session_state.index_data
    top_chunks = retrieve_top_chunks(user_input, st.session_state.embedder, index, chunks, metadata)
    assistant_reply = call_groq(st.session_state.history, user_input, top_chunks)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    st.chat_message("assistant").markdown(assistant_reply)
