from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# Load all text files from custom_data/
docs = []
folder_path = "custom_data"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file_name), encoding='utf-8')
        docs.extend(loader.load())

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Generate embeddings and save vector index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("✅ Ingestion complete: your custom data is ready!")
