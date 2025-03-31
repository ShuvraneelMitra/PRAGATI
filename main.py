import gradio.utils
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.checker import parse_claims, generate_query, search_web, verify_claim, should_continue
from agents.states import QuestionState
from agents.persona import qgen_graph
# from agents.answer import agen_graph
import logging
import os
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.states import FactCheckerState
from agents.schemas import FRPair
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_fact_checker_graph() -> StateGraph:
    """Create the fact checker workflow graph"""
    # defining the nodes
    workflow = StateGraph(FactCheckerState)
    workflow.add_node("parse_claims", parse_claims)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("search_web", search_web)
    workflow.add_node("verify_claim", verify_claim)
    # Define the edges
    workflow.add_edge(START, "parse_claims")
    workflow.add_edge("parse_claims", "generate_query")
    workflow.add_edge("generate_query", "search_web")
    workflow.add_edge("search_web", "verify_claim")
    workflow.add_conditional_edges("verify_claim", should_continue)
    # Compile the graph
    compiled_graph = workflow.compile()

    return compiled_graph

# def create_graph(filepath: gradio.utils.NamedString) -> CompiledStateGraph:
#     graph_builder = StateGraph(QuestionState)
#     graph_builder.add_node("question", qgen_graph())
#     graph_builder.add_node("answer", agen_graph())
#     graph_builder.add_edge(START, "question")
#     graph_builder.add_edge("question", "answer")

#     graph = graph_builder.compile()
#     return graph

def fact_check(text: str) -> List[FRPair]:
    """Run the fact checking process on a text input"""
    logger.info("Starting fact checking process")
    checker = create_fact_checker_graph()
    final_state = checker.invoke({"inputs": text})

    return final_state["pairs"]


def format_results(pairs: List[FRPair]) -> str:
    """Format the fact checking results into a readable string"""
    results = []
    for i, pair in enumerate(pairs):
        result = f"Claim {i+1}: {pair.claim}\n"
        if hasattr(pair, "verification") and pair.verification:
            v = pair.verification
            if v.get("score", "N/A") == 0:
                result += "Claim is Unverified\n"
            else:
                result += f"Score: {v.get('score', 'N/A')}/5\n"
        else:
            result += "Verification failed\n"
        results.append(result)

    return "\n" + "-" * 50 + "\n".join(results)


# Example usage
if __name__ == "__main__":
    while(input("Enter q to quit: ") != 'q'):
        input_text = input("Enter text to fact check: ")
        results = fact_check(input_text)
