from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_vector_store(documents, persist_directory="vectorstore/faiss_index"):
    # Load HF embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index
    vectordb = FAISS.from_documents(documents, embedding_model)
    
    # Persist the index to disk
    vectordb.save_local(persist_directory)
    return vectordb

def load_vector_store(persist_directory="vectorstore/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    return vectordb
