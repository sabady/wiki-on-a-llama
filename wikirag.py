import os
import boto3
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
DATA_DIR = "/home/shany/Downloads/wiki/DevOps"  
CHROMA_DB_DIR = "./chroma_db"  
OLLAMA_MODEL = "llama3"  
BUCKET_NAME = "belong-wiki-bucket"

# Ensure model is available
os.system(f'ollama pull {OLLAMA_MODEL}')  # Local model name

# Step 1: Load Markdown Files from S3
def download_markdown_from_s3(BUCKET_NAME, local_directory):
    s3 = boto3.client("s3")
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    def download_recursive(bucket, prefix=""):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".md"):
                        relative_path = obj["Key"].lstrip(prefix)
                        local_path = os.path.join(local_directory, relative_path)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        s3.download_file(bucket, obj["Key"], local_path)
                        print(f"Downloaded {obj['Key']} to {local_path}")

    download_recursive(BUCKET_NAME)

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
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Execution
if __name__ == "__main__":
#    print("Downloading Markdown files from S3...")
#    download_markdown_from_s3(BUCKET_NAME, DATA_DIR)
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
    retriever.search_kwargs["k"] = 7
    
    print("Initializing RAG chain...")
    rag_chain = setup_rag_chain(retriever)
    
    print("RAG system ready! Type your query.")
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = rag_chain.invoke({"query": query})
        response_text = response.get("result", "No response generated.")
        print("\033[1m" + "Answer: " + response_text + "\033[0m")

