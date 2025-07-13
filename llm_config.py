from langchain.chat_models import ChatOpenAI
import os

def load_llm():
    return ChatOpenAI(
        model="mistralai/mistral-7b-instruct", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

