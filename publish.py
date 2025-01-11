from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from parser.parser import PDFTextExtractor
import pandas as pd
from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

def invoke_llm_langchain(messages, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=5000):
    """
    Invoke the LLM with the given messages and handle response safely.
    """
    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    try:
        response = llm.invoke(messages)
        content = response.content
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
        messages.append(AIMessage(content=content))
        return messages, input_tokens, output_tokens
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return messages, 0, 0

def prompt_for_checking(paper_json):
    system_message = (
        "You are a reviewer for research papers. Judge whether a research paper "
        "is publishable or not using the provided content."
    )
    prompt = (
        f"Consider the abstract, introduction, and conclusion of this paper. "
        f"Predict whether it is publishable or not based on these: {paper_json}. "
        f"Respond only with either 'Publishable' or 'Not Publishable'."
    )
    return system_message, prompt

def guidelines_check():
    df = pd.read_csv('/mnt/c/users/nabay/OneDrive/Desktop/Publishability-of-Research/guidelines.csv')
    gd = []
    for gdlines in df["Formatted by GPT"]:
        gd.append(gdlines)
    text = " ".join(gd)
    return text

class ResearchReviewAgent:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.model = model
        self.tools = [
            self.guidelines_check,  # Use self.guidelines_check here
            self.invoke_llm_tool
        ]
        self.llm_with_tools = model.bind_tools(self.tools)
        self.graph = lg.Graph(name="Research Paper Review Process")

    def guidelines_check(self):
        # This method is now properly part of the class
        df = pd.read_csv('/mnt/c/users/nabay/OneDrive/Desktop/Publishability-of-Research/guidelines.csv')
        gd = []
        for gdlines in df["Formatted by GPT"]:
            gd.append(gdlines)
        text = " ".join(gd)
        return text

    def invoke_llm_tool(self, paper_content):
        system_message, prompt = prompt_for_checking(paper_content)
        messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
        messages, input_tokens, output_tokens = invoke_llm_langchain(messages, model=self.model)
        return messages[-1].content

    def review_paper(self, paper_path):
        # Create nodes in the graph for each stage
        start_node = self.graph.add_node("Start")
        guidelines_node = self.graph.add_node("Guidelines Check")
        llm_invoke_node = self.graph.add_node("Invoke LLM for Review")
        paper_review_node = self.graph.add_node("Paper Review Decision")

        # Add edges to define the flow
        self.graph.add_edge(start_node, guidelines_node)
        self.graph.add_edge(guidelines_node, llm_invoke_node)
        self.graph.add_edge(llm_invoke_node, paper_review_node)

        # Extract paper content and check guidelines
        extractor = PDFTextExtractor(paper_path)
        extractor.analyze_fonts()
        extractor.tag_text(False)
        extractor.save_as_json('parsed.json')
        paper_json = extractor.extract_to_json()  # Assuming extract_to_json outputs the necessary abstract, intro, conclusion
        guideline_text = self.guidelines_check()  # Correctly use self.guidelines_check
        print("Guidelines for publishability:")
        print(guideline_text)
        decision = self.invoke_llm_tool(paper_json)
        self.graph.add_edge(paper_review_node, f"Decision: {decision}")
        return decision

    def show_graph(self):
        # Visualize the LangGraph
        self.graph.visualize()

# Example usage
if __name__ == "__main__":
    agent = ResearchReviewAgent()
    paper_path = '/mnt/c/users/nabay/OneDrive/Desktop/Publishability-of-Research/KDSH_2025_Dataset/Reference/Non-Publishable/R001.pdf'  # Update with actual file path
    result = agent.review_paper(paper_path)
    print(f"The paper is deemed: {result}")
    agent.show_graph()  # Show the graph at the end

