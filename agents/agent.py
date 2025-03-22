from langchain_core.messages import AIMessage, HumanMessage
from utils.chat import invoke_llm_langchain

if __name__ == "__main__":
    messages = [HumanMessage(content="Why attention is all you need is such an epic paper?")]
    response, _, _ = invoke_llm_langchain(messages)
    print("\n ALLAH:\n")
    print(response[-1].content)  