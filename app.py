import os
import uuid
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- Utilities ----------
def file_md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def ensure_folder(p: str):
    os.makedirs(p, exist_ok=True)

UPLOAD_DIR = "uploads"
ensure_folder(UPLOAD_DIR)

# ---------- Cache the LLM once (global) ----------
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "google/flan-t5-base"):
    # GPT-2 is tiny but not instruction-tuned; for better Q/A try "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)

    llm_pipe = pipeline(
        "text2text-generation",          # if you switch to flan-t5-base, use "text2text-generation"
        model=model,
        tokenizer=tokenizer,
        device=-1,                  # CPU (set 0 for GPU)
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=llm_pipe)

# ---------- Cache the RETRIEVER per file hash ----------
@st.cache_resource(show_spinner=True)
def build_retriever_for_file(file_path: str, file_hash: str):
    """
    Cache key is (file_path, file_hash). If contents change, file_hash changes,
    so we build a new vector store. If same file is uploaded again, cache is reused.
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Optional: make the source name human-friendly and stable
    for d in docs:
        d.metadata["source_name"] = os.path.basename(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embeddings)
    return vs.as_retriever()

# ---------- Build a chain (not cached) ----------
def make_rag_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ---------- UI ----------
st.set_page_config(page_title="📄 Personalized RAG Assistant", page_icon="📄")
st.title("📄 Personalized RAG Assistant")

with st.sidebar:
    st.caption("Model")
    model_choice = st.selectbox(
        "Choose model",
        ["openai-community/gpt2", "google/flan-t5-base"],  # add more if you like
        index=0
    )
    if st.button("Clear all caches"):
        st.cache_resource.clear()
        st.experimental_rerun()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    # Read bytes once; use bytes hash as cache key (works across users)
    bytes_data = uploaded.read()
    file_hash = file_md5_bytes(bytes_data)

    # Save with a unique filename so metadata shows the real file
    unique_name = f"{uuid.uuid4()}_{uploaded.name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(bytes_data)

    # Load resources
    with st.spinner("Loading model..."):
        llm = load_llm(model_choice)

    with st.spinner("Indexing your document..."):
        retriever = build_retriever_for_file(save_path, file_hash)

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            rag = make_rag_chain(retriever, llm)
            result = rag.invoke({"query": query})

        st.subheader("🤖 Answer")
        st.write(result["result"])

        st.subheader("📚 Source(s)")
        for i, doc in enumerate(result["source_documents"], start=1):
            src = doc.metadata.get("source_name") or doc.metadata.get("source", "Unknown")
            st.write(f"{i}. {src}")
        st.caption(f"Current file hash: `{file_hash}`")

else:
    st.info("Upload a PDF to get started.")
