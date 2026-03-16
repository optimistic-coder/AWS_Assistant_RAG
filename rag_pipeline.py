# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline — glacier.pdf chatbot
# Run each section top-to-bottom in Google Colab.
# Switch LLM: set USE_GEMINI = True and add your Google AI Studio key.
# ─────────────────────────────────────────────────────────────────────────────

USE_GEMINI = False          # ← set True to use Gemini Flash (fast, free API)
GOOGLE_API_KEY = ""         # ← paste your key from aistudio.google.com
PDF_PATH = "glacier.pdf"    # ← filename as uploaded to Colab /content/
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

# ── STAGE 1 — Install & Load ──────────────────────────────────────────────────
# !pip install -r requirements.txt -q
# Or run these lines individually:
# !pip install langchain langchain-community pypdf langchain-text-splitters -q
# !pip install sentence-transformers torch chromadb langchain-chroma -q
# !pip install transformers accelerate langchain-huggingface -q   # local LLM
# !pip install langchain-google-genai -q                          # Gemini Flash

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[Stage 1] Loaded {len(docs)} page(s) from {PDF_PATH}")
print(f"          Metadata sample: {docs[0].metadata}")


# ── STAGE 2 — Chunking ────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False,
    add_start_index=True,
)

chunks = splitter.split_documents(docs)
print(f"[Stage 2] {len(chunks)} chunks — avg {int(sum(len(c.page_content) for c in chunks)/len(chunks))} chars each")


# ── STAGE 3 — Embedding ───────────────────────────────────────────────────────
import torch
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

texts = [c.page_content for c in chunks]
vectors = np.array(embed_model.embed_documents(texts))
print(f"[Stage 3] Embeddings shape: {vectors.shape} on {device}")


# ── STAGE 4 — Vector Store ────────────────────────────────────────────────────
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    collection_name="glacier_docs",
    persist_directory="./chroma_db",
)
print(f"[Stage 4] Indexed {vectorstore._collection.count()} vectors in ChromaDB")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)
test = retriever.invoke("glacier retreat")
print(f"          Retriever test: {len(test)} chunks returned")


# ── STAGE 5 — LLM Setup ───────────────────────────────────────────────────────
if USE_GEMINI:
    # ── Option B: Gemini Flash (fast, free, recommended for chatbot) ──────────
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        streaming=True,
    )
    print("[Stage 5] Using Gemini 1.5 Flash via Google AI API")
else:
    # ── Option A: Gemma-2 2B local (slow on CPU, needs HF token + license) ────
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    HF_TOKEN = ""   # ← huggingface.co/settings/tokens
    login(HF_TOKEN)

    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"[Stage 5] Gemma-2 2B loaded on {device}")


# ── STAGE 6 — RAG Chain ───────────────────────────────────────────────────────
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Clean prompt — works for both Gemini and Gemma (no turn tags needed for Gemini)
if USE_GEMINI:
    template = """You are a helpful assistant.
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
else:
    template = """<start_of_turn>user
You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}<end_of_turn>
<start_of_turn>model
"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(
        f"[p.{d.metadata['page']}] {d.page_content}" for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=retriever,
)

print("[Stage 6] RAG chain ready")


# ── QUERY — Single question ────────────────────────────────────────────────────
question = "How fast are glaciers retreating and what is the main cause?"

if USE_GEMINI:
    # Streaming — tokens appear as they generate
    print("\nAnswer: ", end="")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()
else:
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}")

# Sources
sources = retriever.invoke(question)
print(f"\nSources: {[f'p.{d.metadata[\"page\"]}' for d in sources]}")


# ── INTERACTIVE LOOP ──────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("Chatbot ready. Type your question or 'exit' to quit.")
print("─"*60)

while True:
    q = input("\nYou: ").strip()
    if not q:
        continue
    if q.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break

    if USE_GEMINI:
        print("Bot: ", end="")
        for chunk in rag_chain.stream(q):
            print(chunk, end="", flush=True)
        print()
    else:
        res = rag_with_sources.invoke(q)
        print(f"Bot: {res['answer']}")

    src = retriever.invoke(q)
    print(f"     [sources: {', '.join(f'p.{d.metadata[\"page\"]}' for d in src)}]")
