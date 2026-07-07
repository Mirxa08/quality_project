# quality_project

Policy Encyclopedia — SOP retrieval and compliance assistant

Quality Project is a Python repository implementing a Policy Encyclopedia: a searchable assistant for hospital Standard Operating Procedures (SOPs) and policies. It extracts text from PDFs, cleans and chunks documents, builds an embedding index (FAISS), and provides a Streamlit-based chat UI that retrieves relevant policy clauses and uses an LLM backend to answer case-specific compliance questions using only the retrieved clauses.

## Key functionality (what the project does)
- Extracts text and metadata from PDF SOP/policy documents into plain text.
- Cleans extracted text to remove headers/footers and standard noise so retrieval is more accurate.
- Splits cleaned documents into overlapping chunks for dense retrieval.
- Generates embeddings for chunks and builds a FAISS index for fast nearest-neighbor search.
- Streamlit UI ("Policy Encyclopedia") to accept case-based queries, preprocess/refine queries, retrieve top matching chunks, and call an LLM-based assistant to produce concise, citation-based answers referencing policy codes and documents.
- Displays the policies/clauses used to form the response for auditing and traceability.

## Features
- End-to-end pipeline: PDF -> text -> cleaned text -> chunks -> embeddings -> FAISS index -> retrieval -> LLM answer.
- Designed for hospital QA, audits, investigative workflows and compliance teams.
- Citation-first answers: assistant is instructed to base responses only on provided clauses and to include policy codes and document names.
- Lightweight embedding model (sentence-transformers) and FAISS for efficient search.

## Repository files (high level)
- `app.py` — Streamlit application (UI + retrieval + LLM calls).
- `extract.py` — PDF extraction (uses PyMuPDF / fitz) to export text and metadata.
- `cleaning.py` — Text cleaning routines to remove boilerplate, titles, page headers, etc.
- `chunk_index.py` — Text chunking, embedding generation and FAISS index creation.
- `llm.py` — Example CLI usage and LLM call wrapper (alternative to streamlit flow).
- `requirements.txt` — primary Python package list.
- `sop_index/` — (output directory) expects FAISS index file, chunk text JSON, and chunk metadata JSON.
- `Clean/` — (output directory) cleaned text files from `cleaning.py`.

## Dependencies / packages used
Primary packages used by code in this repository (from `requirements.txt` and imports):
- faiss-cpu — FAISS index and similarity search
- sentence-transformers — embedding model (e.g., `all-MiniLM-L6-v2`)
- streamlit — web UI
- groq — client used to call the Groq LLM service in examples
- numpy — numerical arrays and type conversions
Additionally required by some scripts:
- PyMuPDF (`fitz`) — PDF text extraction (used in `extract.py`)
- (dev/test) pytest, flake8, black — recommended for development (not required at runtime)

Note: confirm the exact package names/versions you need in `requirements.txt` (some packages may have platform-specific wheels; e.g., faiss-cpu).

## Configuration
- Streamlit secrets: `app.py` reads the LLM API key from Streamlit secrets (e.g., `st.secrets["api"]["groq_key"]`). You can also configure the API key via environment variables in non-Streamlit contexts as demonstrated in `llm.py`.
- File paths:
  - PDFs (input) → configured in `extract.py` (PDF_DIR / OUTPUT_DIR).
  - Cleaned text output → `Clean/` (used by `chunk_index.py`).
  - Index & metadata → `sop_index/faiss.index`, `sop_index/chunks.json`, `sop_index/chunk_metadata.json`.

## Installation (example)
1. Clone the repository:
   git clone https://github.com/Mirxa08/quality_project.git
   cd quality_project

2. Recommended: create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .\.venv\Scripts\activate   # Windows (PowerShell)

3. Install runtime dependencies:
   pip install -r requirements.txt
   # If PyMuPDF not listed, install it if needed:
   pip install PyMuPDF

4. If FAISS installation fails for your platform, see FAISS installation docs for alternatives or wheels.

## Typical usage / pipeline
1. Extract text from PDFs
   - Configure input/output paths in `extract.py`, then run:
     python extract.py
   - Output: `.txt` files and `metadata.json` in the configured output directory.

2. Clean extracted text
   - Configure `INPUT_DIR` and `OUTPUT_DIR` in `cleaning.py`, then run:
     python cleaning.py
   - Output: cleaned `.txt` files in `Clean/`.

3. Chunk & build FAISS index
   - Ensure `Clean/` contains cleaned text files, then:
     python chunk_index.py
   - Output: `sop_index/faiss.index`, `sop_index/chunks.json`, `sop_index/chunk_metadata.json`.

4. Run the Streamlit app (Policy Encyclopedia)
   - Make sure `sop_index/` files exist and the LLM API key is configured (Streamlit secrets or env).
   - Start the UI:
     streamlit run app.py

## File structure (example)
- app.py
- llm.py
- extract.py
- cleaning.py
- chunk_index.py
- requirements.txt
- Clean/                 # cleaned .txt outputs (created by cleaning.py)
- sop_index/             # faiss.index, chunks.json, chunk_metadata.json

## Development notes
- Embedding model: configured to `all-MiniLM-L6-v2` (sentence-transformers) in multiple scripts.
- Retrieval: top K = 5 by default; adjust `TOP_K` constants as needed.
- LLM prompts: the app includes strict system prompts instructing the assistant to only use retrieved clauses and to cite policy codes/document names for traceability.

## Testing
- Add tests under a `tests/` directory using pytest. Tests could cover:
  - Extraction pipeline correctness (small fixture PDFs → text)
  - Cleaning logic (text cleanup unit tests)
  - Chunking behavior (length/overlap expectations)
  - Retrieval end-to-end (mock embeddings or small dataset)

## License
If you'd like a license, add a `LICENSE` file (e.g., MIT or Apache-2.0).

## Maintainer / Contact
Maintainer: Mirxa08  
For questions or help, open an issue in this repository.
```
