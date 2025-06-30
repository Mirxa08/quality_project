import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# Constants
INDEX_FILE = "./sop_index/faiss.index"
CHUNKS_FILE = "./sop_index/chunks.json"
META_FILE = "./sop_index/chunk_metadata.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# API Setup
api_key = st.secrets["api"]["groq_key"]
client = Groq(api_key=api_key)

# ========== Cacheable Resources ========== #
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_index_and_data():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, chunks, metadata

# ========== Smart Preprocessor ========== #
def refine_query(user_input):
    system_prompt = (
        "You are a query preprocessing assistant for a hospital SOP retrieval system.\n"
        "Your job is to analyze the user’s input and do ONE of the following:\n\n"
        "1. If the input is a case-based query meant for SOP or clause lookup (e.g. incident, compliance, policy lookup), rewrite it to make it more suitable for retrieval.\n"
        "2. If the input is follow-up discussion, clarification, or a reasoning request (e.g. asking why something is omitted or asking for elaboration), return it AS IS.\n\n"
        "NEVER change the meaning.\n"
        "NEVER hallucinate content.\n"
        "You must reply with only the final version of the query that should be used downstream.\n\n"
        f"User input: {user_input}"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.1,
        max_tokens=1000
    )
    st.toast(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

# ========== FAISS Retrieval ========== #
def retrieve_top_chunks(query, embedder, index, chunks, metadata, top_k=TOP_K):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    seen_keys = set()
    for i in I[0]:
        if i == -1 or i >= len(chunks):
            continue
        key = f"{metadata[i]['source']}_chunk_{metadata[i]['chunk_index']}"
        if key not in seen_keys:
            seen_keys.add(key)
            results.append({
                "text": chunks[i],
                "source": metadata[i]["source"],
                "chunk_index": metadata[i]["chunk_index"]
            })
    return results

# ========== Main LLM Answering Agent ========== #
def call_groq(history, query, context_chunks):
    context_text = "\n\n".join(
        f"[{c['source']} | Chunk {c['chunk_index']}]:\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "You are a hospital SOP compliance assistant AI. Your purpose is to analyze a user query and answer it using only the information provided in the contextual chunks below.\n\n"
        "Guidelines:\n"
        "- Use ONLY the clauses from the provided chunks. Do NOT invent or guess any content not explicitly mentioned.\n"
        "- You must cite the clause code (e.g., 4.3.3) and policy document name (e.g., IHHN/ALL/MD/CORE/POL/PTP/2022/V02) for every statement you make in bullets. Avoid writing Chunk numbers (Chunk #)\n"
        "- If multiple clauses are relevant, cite each one clearly and briefly state its relevance.\n"
        "- If none of the chunks are relevant to the user's question, respond with: 'I could not find relevant clauses for this case in the available policies. Please refine your query or specify a document.'\n"
        "- Be concise, accurate, and strictly based on provided data.\n"
        "- You may be asked follow-up questions. Maintain previous context."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.4,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

# ========== Streamlit UI ========== #
st.set_page_config(page_title="Policy Encyclopedia", layout="wide")
st.title("🏥 Policy Encyclopedia")

if "welcome_shown" not in st.session_state:
    st.toast("💡 Ask a case-related question to find relevant policies to that scenario!", icon="💡")
    st.session_state.welcome_shown = True

with st.expander("ℹ️ About this Assistant"):
    st.markdown("""
    **Welcome to the Policy Encyclopedia!**  
    - Enter a **case scenario or compliance question**.  
    - The system will search SOP documents and return **relevant clauses**.  
    - You can continue the conversation for more details.  
    - Click **🆕 New Case** to start over.

    ✅ Designed for hospital QA, audits, and investigative workflows.
    """)

# Session State Init
if "history" not in st.session_state:
    st.session_state.history = []

if "embedder" not in st.session_state:
    st.session_state.embedder = get_embedder()

if "index_data" not in st.session_state:
    st.session_state.index_data = load_index_and_data()

# New Case Reset
if st.button("🆕 New Case"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Show Chat History
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Input
user_input = st.chat_input("Enter your SOP-related question...")

if user_input:
    st.chat_message("user").markdown(user_input)

    refined_query = refine_query(user_input)
    index, chunks, metadata = st.session_state.index_data
    top_chunks = retrieve_top_chunks(refined_query, st.session_state.embedder, index, chunks, metadata)
    assistant_reply = call_groq(st.session_state.history, user_input, top_chunks)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    st.chat_message("assistant").markdown(assistant_reply)

    with st.expander("📎 Policies Used"):
        for c in top_chunks:
            st.markdown(f"**{c['source']}**")
