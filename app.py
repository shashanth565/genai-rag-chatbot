import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv

# LangChain / OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain

# =========================
# Env & page config
# =========================
# --- Custom background (gradient) ---
st.markdown("""
<style>
/* Page background */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, #0ea5e922, transparent 60%),
              radial-gradient(1000px 500px at 90% 20%, #22c55e22, transparent 60%),
              linear-gradient(180deg, #0B1220, #0B1220);
}
/* Card/panel background (keep readable) */
.block-container { padding-top: 2rem; }
.css-1y4p8pa, .st-emotion-cache-13ln4jf, .st-emotion-cache-1r6slb0 { 
  background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()  # loads .env locally
# If running on Streamlit Cloud, allow reading from Secrets
if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="GenAI Q&A (RAG)", page_icon="💬", layout="wide")
st.title("💬 GenAI Q&A Chatbot (RAG)")

# Small debug panel (does not show your key)
with st.expander("Debug (temporary)"):
    st.write("OPENAI_API_KEY set?", bool(os.getenv("OPENAI_API_KEY")))

# =========================
# Helpers
# =========================
INDEX_DIR = "faiss_index"
CUSTOM_DIRS = ["custom_data", "uploads"]

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def load_files_from_folder(folder: str):
    """Load all .txt and .pdf from a folder into LangChain Documents."""
    docs = []
    p = Path(folder)
    if not p.exists():
        return docs
    for path in p.glob("**/*"):
        if path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            try:
                docs.extend(PyPDFLoader(str(path)).load())
            except Exception:
                # Fallback to pdfplumber for tricky PDFs
                docs.extend(pdfplumber_load(str(path), source_name=path.name))
    return docs

def pdfplumber_load(pdf_path: str, source_name: str = "uploaded.pdf"):
    """Fallback: extract text from a PDF using pdfplumber."""
    out = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    out.append(Document(page_content=txt, metadata={"source": source_name, "page": i}))
    except Exception as e:
        st.warning(f"pdfplumber failed on {source_name}: {e}")
    return out

def load_uploaded_files(uploaded_files):
    """Return Documents from uploaded PDFs/TXTs (temporary files)."""
    all_docs = []
    for uf in uploaded_files:
        name = uf.name.lower()
        data = uf.read()

        if name.endswith(".txt"):
            with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(data)
                path = tmp.name
            all_docs.extend(TextLoader(path, encoding="utf-8").load())
            os.remove(path)

        elif name.endswith(".pdf"):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                path = tmp.name

            # Try PyPDFLoader first
            docs = []
            try:
                docs = PyPDFLoader(path).load()
            except Exception:
                docs = []

            # Fallback to pdfplumber if empty/failed
            if not docs or all(not d.page_content.strip() for d in docs):
                docs = pdfplumber_load(path, source_name=uf.name)

            all_docs.extend(docs)
            os.remove(path)

        else:
            st.warning(f"Unsupported file type: {uf.name}. Use .pdf or .txt")

    # Filter totally empty pages
    return [d for d in all_docs if getattr(d, "page_content", "").strip()]

def build_vectorstore_from_docs(docs, label="Building…"):
    """Create FAISS index from Docs (with guards, progress and errors)."""
    docs = [d for d in docs if getattr(d, "page_content", "").strip()]
    if not docs:
        st.error("No extractable text found in the provided files/folders. "
                 "If your PDF is scanned, convert it to a searchable PDF or try a .txt export.")
        return None

    chunks = split_docs(docs)
    if not chunks:
        st.error("Could not create chunks from the provided documents.")
        return None

    with st.status(label, expanded=True) as status:
        try:
            st.write(f"Chunks: {len(chunks)} → creating embeddings…")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = FAISS.from_documents(chunks, embeddings)
            status.update(label="Done ✅", state="complete")
            return db
        except Exception as e:
            status.update(label=f"Failed ❌: {e}", state="error")
            st.error(f"Embedding error: {e}")
            return None

def save_uploads(uploaded_files, target_folder="uploads"):
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        dest = Path(target_folder) / uf.name
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved.append(str(dest))
    return saved

# =========================
# Sidebar (explicit actions)
# =========================
st.sidebar.header("Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload .pdf / .txt (multiple allowed)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="uploader",
)
persist_uploads = st.sidebar.checkbox("Save uploads to disk (reusable)", value=True)
use_local = st.sidebar.checkbox("Use local indexes/folders", value=True)

build_from_uploads_btn = st.sidebar.button("Build index from uploaded files")
rebuild_local_btn = st.sidebar.button("Rebuild local FAISS index")

# Prefer in-memory index if created
vectorstore = st.session_state.get("in_memory_db", None)

# A) Build from uploads (only on button)
if build_from_uploads_btn:
    try:
        if not uploaded_files:
            st.sidebar.warning("Upload at least one PDF/TXT first.")
        else:
            if persist_uploads:
                # Save uploads → combine folders → rebuild local index (persistent)
                save_uploads(uploaded_files, "uploads")
                all_docs = []
                for d in CUSTOM_DIRS:
                    all_docs.extend(load_files_from_folder(d))
                db = build_vectorstore_from_docs(all_docs, label="Rebuilding local index from uploads + folders…")
                if db:
                    db.save_local(INDEX_DIR)
                    st.sidebar.success("Uploads saved & FAISS index updated.")
                    # Also set as active for this session
                    st.session_state["in_memory_db"] = db
                    vectorstore = db
            else:
                # In-memory only for this session
                docs = load_uploaded_files(uploaded_files)
                db = build_vectorstore_from_docs(docs, label="Building in-memory index from uploads…")
                if db:
                    st.session_state["in_memory_db"] = db
                    vectorstore = db
                    st.sidebar.success("In-memory index ready (this session).")
    except Exception as e:
        st.sidebar.error(f"Build from uploads failed: {e}")

# B) Rebuild local FAISS index (manual)
if rebuild_local_btn:
    try:
        all_docs = []
        for d in CUSTOM_DIRS:
            all_docs.extend(load_files_from_folder(d))
        db = build_vectorstore_from_docs(all_docs, label="Rebuilding local FAISS index…")
        if db:
            db.save_local(INDEX_DIR)
            st.sidebar.success("Local FAISS index rebuilt.")
            st.session_state["in_memory_db"] = db
            vectorstore = db
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

# C) Load local persistent index if no in-memory is set and toggle is ON
if vectorstore is None and use_local:
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        st.sidebar.info("Using local FAISS index (custom_data/ + uploads/).")
    except Exception:
        st.sidebar.info("No local FAISS index loaded. Use the buttons above to build one.")

# =========================
# Chat UI with history
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []         # for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # for CRC: list[(user, assistant)]

model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini"])
llm = ChatOpenAI(model=model_name)

st.subheader("Chat")
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a question about your documents…")

# Small debug state
with st.expander("Debug (state)"):
    st.write("Has in_memory_db?", "in_memory_db" in st.session_state)
    st.write("Vectorstore loaded?", vectorstore is not None)

if prompt:
    try:
        if vectorstore is None:
            st.error("No data loaded. Upload and build the index (buttons in the sidebar) first.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append(("user", prompt))

            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

            with st.spinner("Thinking…"):
                result = crc({"question": prompt, "chat_history": st.session_state.chat_history})
                answer = result.get("answer", "(No answer)")

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append(("assistant", answer))
            st.session_state.chat_history.append((prompt, answer))

            # Show sources (best-effort)
            st.caption("— Top matches (sources) —")
            try:
                hits = vectorstore.similarity_search(prompt, k=4)
                for i, d in enumerate(hits, 1):
                    src = d.metadata.get("source", "uploaded/unknown")
                    page = d.metadata.get("page", None)
                    tail = f" (page {page})" if page is not None else ""
                    st.caption(f"{i}. {src}{tail}")
            except Exception as e:
                st.caption(f"(Could not fetch sources: {e})")
    except Exception as e:
        st.error(f"Error while answering: {e}")
