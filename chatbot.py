import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List, Any
import torch

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Local RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🚀 Privacy-First PDF Chatbot")
st.markdown("**Fully local RAG pipeline — no API keys, no cloud, no data leaks.**")
st.divider()

# ─── Session State ──────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **LangChain** — RAG pipeline
    - **FAISS** — Vector search
    - **Sentence Transformers** — Embeddings
    - **HuggingFace** — Local LLM (Flan-T5)
    - **pypdf** — PDF parsing
    - **Streamlit** — UI
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
    top_k = st.slider("Top K Retrievals", 1, 5, 3)

# ─── Helper: Extract Text from PDF ─────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# ─── Helper: Build Vector Store ────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Building vector store...")
def build_vectorstore(text: str, _chunk_size: int, _chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ─── Custom LLM Wrapper (fixes text2text-generation pipeline error) ────────────
@st.cache_resource(show_spinner="🤖 Loading local LLM (this may take a minute)...")
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    class FlanT5LLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "flan-t5"

        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return FlanT5LLM()

# ─── Build QA Chain ────────────────────────────────────────────────────────────
def build_qa_chain(vectorstore, llm, top_k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ─── Process PDF on Upload ─────────────────────────────────────────────────────
if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        st.error("⚠️ Could not extract text from the PDF. Try a different file.")
    else:
        st.success(f"✅ PDF loaded! Extracted **{len(raw_text):,}** characters.")

        vectorstore = build_vectorstore(raw_text, chunk_size, chunk_overlap)
        llm = load_llm()
        st.session_state.qa_chain = build_qa_chain(vectorstore, llm, top_k)
        st.success("🧠 Chatbot is ready! Ask your questions below.")

# ─── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("## 💬 Chat with your PDF")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_query = st.chat_input("Ask something about your PDF...")

if user_query:
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.qa_chain({"query": user_query})
                answer = result["result"]
                source_docs = result.get("source_documents", [])

            st.write(answer)

            if source_docs:
                with st.expander("📚 Source Chunks Used"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ─── Clear Chat Button ─────────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()