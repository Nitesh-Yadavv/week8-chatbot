# Loan Approval Q&A Chatbot (RAG with LangChain + OpenRouter)

This is an intelligent Q&A chatbot built using **Retrieval-Augmented Generation (RAG)** architecture. It uses **LangChain**, **FAISS**, and a **lightweight LLM via OpenRouter** to answer questions from a loan approval dataset.


---

## Features

- RAG-based question answering
- CSV ingestion and chunking
- FAISS vector search
- Supports free OpenRouter API (Mistral, LLaMA3, Claude Haiku, etc.)
- Runs via a Streamlit web app

---

##  How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/loan-rag-chatbot.git
cd loan-rag-chatbot
```
### 2.Install Dependencies
```bash
pip install -r requirements.txt
```
- It's recommended to use a virtual environment

### 3.Add `.env` file
- Create a .env file in root with:

```bash
OPENROUTER_API_KEY=your_openrouter_key
```

- You can get a key from: https://openrouter.ai/keys (Sign in → Copy Key)

### 5. Run the App

```bash
streamlit run app.py
```
- The first run will create the FAISS index. It may take a few seconds.

# Example Questions to Ask
- "What is the average applicant income?"

- "How many people got their loans approved?"

- "Do self-employed applicants get loans more often?"

- "What is the most common education level of approved applicants?"

# Built With
- LangChain

- FAISS

- OpenRouter

- Streamlit

- Pandas


