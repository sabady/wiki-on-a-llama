import os
import markdown
import chromadb
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

# Configuration
DATA_DIR = "/home/shany/Downloads/wiki/DevOps"  # Directory containing markdown files
CHROMA_DB_DIR = "./chroma_db"  # Directory for ChromaDB storage
OLLAMA_MODEL = "wizardlm2"  # Local model name

# Ensure model is available
os.system(f'ollama pull {OLLAMA_MODEL}')  # Local model name

# Step 1: Load Markdown Files
def load_markdown_files(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(page_content=content, metadata={"source": path})
                    documents.append(doc)
    return documents

# Step 2: Create Embeddings and Store in ChromaDB
def create_vector_store(documents):
    embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectordb = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DB_DIR)
    # vectordb.persist()
    return vectordb

# Step 3: Set Up Retriever
def setup_retriever(vectordb):
    return VectorStoreRetriever(vectorstore=vectordb)

# Step 4: Set Up LLM with RetrievalQA
def setup_rag_chain(retriever):
    llm = OllamaLLM(model=OLLAMA_MODEL, max_tokens=500, streaming=True)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Execution
if __name__ == "__main__":
    print("Loading Markdown files...")
    documents = load_markdown_files(DATA_DIR)
    
    print("Creating vector store...")
    if os.path.exists(CHROMA_DB_DIR):
        embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)
        vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    else:
        vectordb = create_vector_store(documents)
    
    print("Setting up retriever...")
    retriever = setup_retriever(vectordb)
    retriever.search_kwargs["k"] = 3
    
    print("Initializing RAG chain...")
    rag_chain = setup_rag_chain(retriever)
    
    print("RAG system ready! Type your query.")
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = rag_chain.invoke({"query": query})
        response_text = response.get("result", "No response generated.")
        print("Answer: " + response_text + "")

