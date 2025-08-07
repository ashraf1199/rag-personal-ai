import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

# Setup model and retriever
@st.cache_resource
def setup_rag(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embed)

    retriever = db.as_retriever()

# Choose the LLM model from Hugging Face
# Change the model name depending on your compute resources:
# - "tiiuae/falcon-rw-1b" or "Qwen/Qwen3-0.6B"    : runs well on low-end GPU/CPU
# - "mistralai/Mistral-7B-Instruct-v0.1" : requires GPU
# - "sshleifer/tiny-gpt2"       : good for quick CPU testing

    model_name = "tencent/Hunyuan-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
# HuggingFace pipeline parameters:
# - max_new_tokens: maximum words to generate per response
# - temperature: randomness of responses (0.7 = balanced)
# - device: set to 0 for GPU, -1 for CPU
    llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1, max_new_tokens=256)

    llm = HuggingFacePipeline(pipeline=llm_pipe)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.title("📄 Personalized RAG Assistant")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    
    rag_chain = setup_rag("temp.pdf")
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Generating answer..."):
            result = rag_chain.invoke({"query": query})
            st.write("### 🤖 Answer:")
            st.write(result["result"])
            
            st.write("### 📚 Source(s):")
            for doc in result["source_documents"]:
                st.write("-", doc.metadata.get("source", "Unknown"))
