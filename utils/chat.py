from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import os
import time

def invoke_llm_langchain(messages, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=5000, api_key=None):
    """
    Invoke the LLM with the given messages
    """
    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    os.environ["GROQ_API_KEY"] = api_key
    net_input = 0
    net_output = 0

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"Error in invoking LLM, sending LLM to sleep for 10 seconds")
        time.sleep(10)
        print(f"Retrying now")
        response = llm.invoke(messages)

    try:
        content = response.content
        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"]
    except Exception as e:
        content = response
        input_tokens = net_input
        output_tokens = net_output

    messages.append(AIMessage(content=content)) 

    return messages, input_tokens, output_tokens