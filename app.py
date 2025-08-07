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

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# ----------- Setup -----------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="📄 Personalized RAG Assistant", page_icon="📄")
st.title("📄 Personalized RAG Assistant")

# ----------- Utils -----------
def file_md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

# ----------- LLM loader (cached) -----------
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    """
    Load a HF model correctly depending on its architecture.
    Works on CPU (Streamlit Cloud) by forcing device=-1.
    """
    try:
        cfg = AutoConfig.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        is_t5_like = cfg.model_type in {"t5", "mt5"} or "t5" in model_name.lower()
        is_seq2seq = is_t5_like or getattr(cfg, "is_encoder_decoder", False)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)

        if is_seq2seq:
            # FLAN / T5 / mT5 / MBART / Pegasus, etc.
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_auth_token=HF_TOKEN,
                torch_dtype="auto",
            )
            task = "text2text-generation"
            gen_kwargs = dict(max_new_tokens=256)
        else:
            # GPT‑2 / Mistral / Llama / Qwen, etc.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=HF_TOKEN,
                torch_dtype="auto",
            )
            task = "text-generation"
            gen_kwargs = dict(max_new_tokens=256, do_sample=True, temperature=0.7)

            # GPT‑2 needs a pad token set to eos to avoid warnings/errors
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                if hasattr(model.config, "pad_token_id"):
                    model.config.pad_token_id = tokenizer.eos_token_id

        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU (Cloud has no GPU)
            **gen_kwargs,
        )

        return HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        raise

# ----------- Build retriever per file hash (cached) -----------
@st.cache_resource(show_spinner=True)
def build_retriever_for_file(file_path: str, file_hash: str):
    """
    Cache key is (file_path, file_hash). If contents change, file_hash changes,
    so we build a new vector store. If the same file is uploaded again, cache is reused.
    """
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Human-friendly source
        base = os.path.basename(file_path)
        for d in docs:
            d.metadata["source_name"] = base

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = FAISS.from_documents(chunks, embeddings)
        return vs.as_retriever()
    except Exception as e:
        st.error(f"❌ Indexing failed: {e}")
        raise

# ----------- Build RAG chain -----------
def make_rag_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ----------- Sidebar -----------
with st.sidebar:
    st.caption("Model")
    model_choice = st.selectbox(
        "Choose model",
        [
            "google/flan-t5-base",     # better Q&A (seq2seq)
            "openai-community/gpt2",   # tiny demo (causal)
        ],
        index=0,
    )

    if st.button("Clear all caches"):
        st.cache_resource.clear()
        st.experimental_rerun()

# ----------- UI: Upload & Ask -----------
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    bytes_data = uploaded.read()
    if not bytes_data:
        st.error("Uploaded file is empty.")
        st.stop()

    file_hash = file_md5_bytes(bytes_data)

    # Save with a unique filename so metadata shows the real file
    unique_name = f"{uuid.uuid4()}_{uploaded.name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(bytes_data)

    with st.spinner("Loading model..."):
        llm = load_llm(model_choice)

    with st.spinner("Indexing your document..."):
        retriever = build_retriever_for_file(save_path, file_hash)

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            rag = make_rag_chain(retriever, llm)
            try:
                result = rag.invoke({"query": query})
            except Exception as e:
                st.error(f"❌ RAG failed: {e}")
                st.stop()

        st.subheader("🤖 Answer")
        st.write(result.get("result", ""))

        st.subheader("📚 Source(s)")
        for i, doc in enumerate(result.get("source_documents", []), start=1):
            src = doc.metadata.get("source_name") or doc.metadata.get("source", "Unknown")
            st.write(f"{i}. {src}")

        st.caption(f"Current file hash: `{file_hash}`")

else:
    st.info("Upload a PDF to get started.")
