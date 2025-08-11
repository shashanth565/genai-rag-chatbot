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
load_dotenv()  # loads .env locally

st.set_page_config(page_title="GenAI Q&A (RAG)", page_icon="💬", layout="wide")

# ---- Secrets (safe for local) ----
def _load_openai_key_from_secrets_if_present():
    """If running on Streamlit Cloud, pull key from st.secrets.
    Wrapped so local runs without secrets.toml don't crash."""
    try:
        if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

def _is_public_mode():
    """Public-safe defaults via Secrets; defaults to False locally."""
    try:
        return st.secrets.get("PUBLIC_APP", "0") == "1"
    except Exception:
        return False

_load_openai_key_from_secrets_if_present()
PUBLIC = _is_public_mode()

def _api_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

# ===== Welcome / Hero =====
st.markdown("""
<div style="padding: 18px 20px; border-radius: 14px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.07);">
  <h2 style="margin:0 0 6px 0;">💬 GenAI Q&A Chatbot (RAG)</h2>
  <p style="margin:0; opacity:0.9">
    Upload PDFs/TXTs or use the local knowledge base. I’ll retrieve the right chunks with FAISS and answer using OpenAI — with sources.
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("👉 How to use (quick start)", expanded=True):
    st.markdown("""
**Option A – Fast test (private, per-session)**
1. In the sidebar: turn **OFF** “Save uploads to disk” and **OFF** “Use local indexes/folders”.
2. Upload a PDF/TXT.
3. Click **Build index from uploaded files**.
4. Ask your question in the chat box.

**Option B – Reusable local index (persists on server)**
1. Turn **ON** “Save uploads to disk” and **ON** “Use local indexes/folders”.
2. Upload files.
3. Click **Rebuild local FAISS index** (this reprocesses everything in `uploads/` + `custom_data/`).
4. Ask questions. Next time you won’t need to re-upload.

**Sample questions**
- “Summarize this document in 5 bullets.”
- “What are the key requirements and dates?”
- “List pros/cons mentioned about &lt;topic&gt;.”
- “Give me a 2-paragraph overview with citations.”
""")

with st.expander("🔐 Privacy & data handling"):
    st.markdown("""
- Each visitor has a **separate session**. Your chat is **not** visible to others.
- With **Save uploads to disk = OFF**, files stay **in memory** for your session only (safer for public demos).
- With **Save uploads to disk = ON**, uploads are written to the server’s `uploads/` folder and included in the shared local index when you rebuild.
- Don’t put secrets in documents. Your **OpenAI API key** is stored as a secret (not in code).
""")

with st.expander("🛠️ Troubleshooting"):
    st.markdown("""
- If the page looks blank after upload, wait for the ✅ status after building the index.
- If you see *“No extractable text”*, your PDF may be scanned — use an OCR/searchable version or paste text into a `.txt`.
- If answers fail: check that **OPENAI_API_KEY** is set; try smaller files; open the **Debug** expander for state.
""")

# Small debug panel (never prints your key)
with st.expander("🐞 Debug"):
    st.write("OPENAI_API_KEY set?", _api_key_present())

# =========================
# Helpers
# =========================
INDEX_DIR = "faiss_index"
CUSTOM_DIRS = ["custom_data", "uploads"]

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

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

def load_files_from_folder(folder: str):
    """Load all .txt and .pdf from a folder into Documents."""
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
                docs.extend(pdfplumber_load(str(path), source_name=path.name))
    return docs

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

            docs = []
            try:
                docs = PyPDFLoader(path).load()
            except Exception:
                docs = []

            if not docs or all(not d.page_content.strip() for d in docs):
                docs = pdfplumber_load(path, source_name=uf.name)

            all_docs.extend(docs)
            os.remove(path)

        else:
            st.warning(f"Unsupported file type: {uf.name}. Use .pdf or .txt")

    return [d for d in all_docs if getattr(d, "page_content", "").strip()]

def build_vectorstore_from_docs(docs, label="Building…"):
    """Create FAISS index from Docs (with guards, progress and errors)."""
    if not _api_key_present():
        st.error("OPENAI_API_KEY not set. Add it to your local .env or Streamlit Secrets.")
        return None

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

# NEW: style helper
def apply_style(prompt: str, style: str) -> str:
    if style == "Detailed":
        return (
            "Answer in a detailed, step-by-step way, include brief reasoning and keep it clear.\n\n"
            f"User question: {prompt}"
        )
    else:
        return (
            "Answer concisely in 2–4 sentences, no fluff.\n\n"
            f"User question: {prompt}"
        )

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

persist_uploads = st.sidebar.checkbox("Save uploads to disk (reusable)", value=not PUBLIC)
use_local = st.sidebar.checkbox("Use local indexes/folders", value=not PUBLIC)

build_from_uploads_btn = st.sidebar.button("Build index from uploaded files")
rebuild_local_btn = st.sidebar.button("Rebuild local FAISS index")

model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])

# NEW: Answer style toggle
answer_style = st.sidebar.radio(
    "Answer style",
    ["Concise", "Detailed"],
    index=0,
    help="Concise = shorter answers. Detailed = more verbose answers."
)

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
                for d in ["custom_data", "uploads"]:
                    all_docs.extend(load_files_from_folder(d))
                db = build_vectorstore_from_docs(all_docs, label="Rebuilding local index from uploads + folders…")
                if db:
                    db.save_local(INDEX_DIR)
                    st.sidebar.success("Uploads saved & FAISS index updated.")
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
        for d in ["custom_data", "uploads"]:
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
        if not _api_key_present():
            st.sidebar.warning("Set OPENAI_API_KEY to load the local index.")
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
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

st.subheader("Chat")
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# Small state debug
with st.expander("Debug (state)"):
    st.write("Has in_memory_db?", "in_memory_db" in st.session_state)
    st.write("Vectorstore loaded?", vectorstore is not None)

prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    try:
        if not _api_key_present():
            st.error("OPENAI_API_KEY not set. Add it to your local .env or Streamlit Secrets.")
        elif vectorstore is None:
            st.error("No data loaded. Upload and build the index (buttons in the sidebar) first.")
        else:
            # Initialize LLM only when needed (avoids errors if key is missing earlier)
            llm = ChatOpenAI(model=model_name)

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append(("user", prompt))

            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

            styled_prompt = apply_style(prompt, answer_style)

            with st.spinner("Thinking…"):
                result = crc({"question": styled_prompt, "chat_history": st.session_state.chat_history})
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

# ===== Download transcript (sidebar) =====
def _format_chat_for_download(messages):
    lines = []
    for role, content in messages:
        who = "You" if role == "user" else "Assistant"
        lines.append(f"{who}: {content}")
    return "\n\n".join(lines)

with st.sidebar.expander("Export", expanded=False):
    if st.session_state.get("messages"):
        transcript = _format_chat_for_download(st.session_state["messages"])
        st.download_button(
            label="⬇️ Download chat (.txt)",
            data=transcript.encode("utf-8"),
            file_name="chat_transcript.txt",
            mime="text/plain",
        )
    else:
        st.caption("Start a chat to enable export.")
