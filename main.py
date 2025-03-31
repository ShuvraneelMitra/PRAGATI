import gradio.utils
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agents.states import QuestionState
from agents.persona import qgen_graph
from agents.answer import agen_graph

def create_graph(filepath: gradio.utils.NamedString) -> CompiledStateGraph:
    graph_builder = StateGraph(QuestionState)
    graph_builder.add_node("question", qgen_graph())
    graph_builder.add_node("answer", agen_graph())
    graph_builder.add_edge(START, "question")
    graph_builder.add_edge("question", "answer")

    graph = graph_builder.compile()
    return graph

if __name__ == "__main__":
    pdf_path = "C:/Users/MITRA/Desktop/Books/Tiny Machine Learning.pdf"
    graph = create_graph(pdf_path)
    print(graph.invoke(QuestionState(messages=[])))