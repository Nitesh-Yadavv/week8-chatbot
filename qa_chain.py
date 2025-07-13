import os
from langchain.chains import RetrievalQA
from loader import load_and_split_loan_data
from embedder import create_vector_store, load_vector_store
from llm_config import load_llm

def setup_qa_chain(data_path="data/Training_Dataset.csv", vectorstore_path="vectorstore/faiss_index"):
    index_path = os.path.join(vectorstore_path, "index.faiss")
    
    # Check if FAISS index exists
    if not os.path.exists(index_path):
        print("No FAISS index found. Creating vector store from CSV...")
        chunks = load_and_split_loan_data(data_path)
        create_vector_store(chunks, vectorstore_path)

    # Load FAISS index
    vectordb = load_vector_store(vectorstore_path)
    llm = load_llm()

    # Build RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
