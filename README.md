# 💬 GenAI Q&A Chatbot (RAG)

Personal document-aware chatbot built with **Streamlit + LangChain + FAISS + OpenAI**.  
Upload PDFs/TXTs, it chunks & embeds them, retrieves relevant context, and answers **with sources**.

### 🔗 Live Demo
https://genai-rag-chatbot-j8z2aw9tmm3idcsg7kf53z.streamlit.app/

---

## ✨ Features

- **Bring your own data**: Upload `.pdf` or `.txt` (multiple files).
- **Retrieval-Augmented Generation (RAG)**: FAISS vector store + similarity search.
- **Citations**: Shows top matches (file name + page).
- **Session privacy**: Each visitor has isolated chat history.
- **Robust PDF parsing**: Falls back to `pdfplumber` if `pypdf` can’t extract text.
- **Public-safe mode**: Optional defaults that **don’t** save uploads on the server.
- **Streamlit UI**: Clean chat, buttons for explicit indexing, debug panels.

---

## 🧱 Architecture

- **UI**: Streamlit (`app.py`)
- **LLM**: `ChatOpenAI` (`gpt-4o-mini` by default)
- **Embeddings**: `text-embedding-3-small`
- **Vector DB**: FAISS (local on disk, optional)
- **Chunking**: RecursiveCharacterTextSplitter (1k chars, 200 overlap)
- **PDF parsing**: `pypdf` → fallback `pdfplumber`

---

## 🚀 Quickstart (Local)

1) Clone and enter the project:
```bash
git clone https://github.com/<your-username>/genai-rag-chatbot.git
cd genai-rag-chatbot

## 🧑‍💻 Author

**Shashanth Reddy Sam Reddy**  
- LinkedIn: https://www.linkedin.com/in/shashanthreddy/
- Email: shashanthreddy7777@gmail.com
