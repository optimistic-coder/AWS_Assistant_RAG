# RAG Pipeline — PDF Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that answers questions
from any PDF document. Built with LangChain, ChromaDB, and HuggingFace — runs
entirely in Google Colab with no paid API required.

---

## What It Does

Upload a PDF → ask questions in plain English → get grounded answers with page citations.

```
PDF  →  Chunks  →  Embeddings  →  ChromaDB  →  LLM  →  Answer + Sources
```

---

## Quick Start (Google Colab)

### 1. Open a new Colab notebook
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Enable GPU runtime *(recommended — 10x faster)*
```
Runtime → Change runtime type → T4 GPU → Save
```

### 3. Upload your PDF
Click the **folder icon** in the left sidebar → drag and drop your PDF file.
The file will be available at `/content/your_file.pdf`.

### 4. Install dependencies
```python
!pip install -r requirements.txt -q
```
Or paste the install commands directly from the notebook cells.

### 5. Run all cells in order
The pipeline is split into 6 stages — run each cell top to bottom.

---

## Project Structure

```
rag_project/
├── rag_pipeline.py      # Full pipeline (all stages combined)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License + third-party notices
└── chroma_db/           # Created at runtime — persisted vector index
```

---

## Pipeline Stages

| Stage | What happens | Key output |
|-------|-------------|------------|
| 1 — Ingestion   | Load PDF pages via PyPDFLoader        | `List[Document]`     |
| 2 — Chunking    | Split into 500-char overlapping chunks | `List[Document]`     |
| 3 — Embedding   | Encode chunks with MiniLM-L6-v2       | `np.array (N, 384)`  |
| 4 — Vector Store| Index into ChromaDB (HNSW, cosine)    | `vectorstore`        |
| 5 — Retrieval   | Top-4 chunks by cosine similarity     | `retriever`          |
| 6 — LLM Query   | Gemma-2 / Gemini Flash answers query  | `str` answer         |

---

## Configuration

### Chunk size
Edit in Stage 2:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # increase for longer context, decrease for precision
    chunk_overlap=100,   # 20% overlap — keeps context across boundaries
)
```

### Number of retrieved chunks
Edit in Stage 4:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

### Switching LLM

**Option A — Gemma-2 2B (local, free, slow on CPU)**
```python
# Requires HuggingFace token + Gemma license acceptance
llm = HuggingFacePipeline(pipeline=pipe)
```

**Option B — Gemini Flash (API, free tier, fast)**
```python
!pip install langchain-google-genai -q
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="YOUR_KEY",   # aistudio.google.com → Get API key
    temperature=0,
    streaming=True,
)
```
> Everything downstream (`retriever`, `rag_chain`) stays identical — only the `llm` line changes.

---

## Streaming Output

Use `.stream()` for a real-time chatbot feel — first token appears in ~0.5s:
```python
for chunk in rag_chain.stream("What causes glacier retreat?"):
    print(chunk, end="", flush=True)
```

---

## Getting API Keys

| Service | Where to get it | Cost |
|---------|----------------|------|
| HuggingFace token | huggingface.co/settings/tokens | Free |
| Gemma-2 access | huggingface.co/google/gemma-2-2b-it → Agree & access | Free |
| Gemini Flash key | aistudio.google.com → Get API key | Free (1M tokens/day) |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NameError: torch not defined` | Missing import | Add `import torch` at top of cell |
| `NameError: retriever not defined` | Kernel restarted | Re-run Stage 4 cell 16 |
| Gemma turn tags in output | Wrong prompt format for Gemini | Remove `<start_of_turn>` tags when using Gemini |
| `chroma_db` not found | Runtime disconnected | Chroma reloads from `./chroma_db` — re-run Stage 4 reload cell |
| Slow responses (30–60s) | Running Gemma on CPU | Switch to Gemini Flash or enable T4 GPU |

---

## Extending This Project

- **Multiple PDFs** — pass a list to `PyPDFLoader` or use `DirectoryLoader`
- **Web pages** — swap `PyPDFLoader` for `WebBaseLoader`
- **Chat history** — wrap `rag_chain` in `RunnableWithMessageHistory`
- **Re-ranking** — add a Cohere or BGE reranker between retriever and LLM
- **Gradio UI** — wrap the Q&A loop in `gr.ChatInterface` for a browser chatbot

---

## License

MIT — see [LICENSE](LICENSE) for full terms and third-party notices.
