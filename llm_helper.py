"""
LLM Helper Module
Initializes the Groq LLM with Llama model
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

def get_llm(model_name: str = None):
    """
    Initialize and return a ChatGroq LLM instance.
    
    Args:
        model_name: Optional model name override. 
                   Defaults to llama-3.3-70b-versatile.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "gsk_your_actual_api_key_here":
        raise ValueError(
            "Please set your GROQ_API_KEY in the .env file. "
            "Get your free API key at https://console.groq.com/"
        )
    
    # Use llama-3.3-70b-versatile as default (current recommended model)
    if model_name is None:
        model_name = "llama-3.3-70b-versatile"
    
    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024
    )
    
    return llm


def get_llm_with_fallback():
    """
    Try primary model first, fallback to alternative if unavailable.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    primary_model = "llama-3.3-70b-versatile"
    fallback_model = "llama-3.1-70b-versatile"
    
    try:
        llm = get_llm(primary_model)
        # Test the connection with a simple call
        return llm
    except Exception as e:
        print(f"Primary model unavailable ({e}), switching to fallback...")
        return get_llm(fallback_model)


if __name__ == "__main__":
    # Test the LLM connection
    try:
        llm = get_llm()
        response = llm.invoke("Say 'Hello, LinkedIn!' in one sentence.")
        print("LLM Test Successful!")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")
