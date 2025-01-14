from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import os

def invoke_llm_langchain(messages, model="llama3-70b-8192", temperature=0.7, max_tokens=5000, api_key=None):
    """
    Invoke the LLM with the given messages
    """
    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    os.environ["GROQ_API_KEY"] = api_key

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"Error in invoking LLM:\n{e}")
        return messages, [], []

    content = response.content
    input_tokens = response.usage_metadata["input_tokens"]
    output_tokens = response.usage_metadata["output_tokens"]

    messages.append(AIMessage(content=content)) 

    return messages, input_tokens, output_tokens