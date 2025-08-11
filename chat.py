from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load your environment variables
load_dotenv()

# Load vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use gpt-4 if you have access

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask questions
while True:
    query = input("Ask something (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print(f"\n🤖 Answer: {answer}\n")
