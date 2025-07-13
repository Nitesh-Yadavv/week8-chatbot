import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_loan_data(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Convert each row into a string
    docs = []
    for i, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(content)

    # Split long text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n", ",", ".", " "]
    )
    chunks = splitter.create_documents(docs)

    return chunks
