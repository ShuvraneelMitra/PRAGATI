from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

def invoke_llm_langchain(messages, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=5000):
    """
    Invoke the LLM with the given messages
    """
    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"Error in invoking LLM:\n{e}")

    content = response.content
    input_tokens = response.usage_metadata["input_tokens"]
    output_tokens = response.usage_metadata["output_tokens"]

    messages.append(AIMessage(content=content)) 

    return messages, input_tokens, output_tokens