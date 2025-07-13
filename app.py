import streamlit as st
from qa_chain import setup_qa_chain
from dotenv import load_dotenv

# Load .env file with OpenAI API key
load_dotenv()

st.set_page_config(page_title="Loan Q&A Chatbot", layout="wide")

st.title("🤖 Loan Approval Q&A Chatbot")
st.markdown("Ask any question about loan approvals based on the training dataset.")

# Initialize session state
if "qa_chain" not in st.session_state:
    with st.spinner("Loading model and vector DB..."):
        st.session_state.qa_chain = setup_qa_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
user_question = st.text_input("Ask your question here:")

if user_question:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke({"query": user_question})
        answer = response["result"]

    st.session_state.chat_history.append((user_question, answer))



# Display chat history
for question, answer in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Bot:** {answer}")
    st.markdown("---")
