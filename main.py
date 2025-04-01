import gradio.utils
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agents.checker import create_fact_checker_graph
from agents.states import QuestionState
from agents.persona import qgen_graph
from agents.answer import agen_graph
import logging
import os
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.states import FactCheckerState
from agents.schemas import FRPair, Paper
from typing import List
from agents.states import CombinedPaperState
import asyncio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_graph(filepath: str) -> CompiledStateGraph:
    init_state = dict(paper=Paper(filepath=filepath, title="Research Paper"))
    graph_builder = StateGraph(QuestionState(**init_state))
    graph_builder.add_node("question", qgen_graph())
    graph_builder.add_node("answer", agen_graph())
    graph_builder.add_edge(START, "question")
    graph_builder.add_edge("question", "answer")

    graph = graph_builder.compile()
    return graph


def fact_check(text: str) -> List[FRPair]:
    """Run the fact checking process on a text input"""
    logger.info("Starting fact checking process")
    checker = create_fact_checker_graph()
    final_state = checker.invoke({"inputs": text},config={"recursion_limit": 10000})

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


def read_and_chunk_file(filepath: str, chunk_size: int = 1024) -> List[str]:
    """Read the file and split it into chunks"""
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
    except UnicodeDecodeError:
        with open(filepath, "rb") as file:
            while True:
                chunk = file.read(chunk_size).decode("utf-8", errors="ignore")
                if not chunk:
                    break
                chunks.append(chunk)
    return chunks


async def process_paper_parallel(paper_filepath: str):
    """
    Process a research paper with both fact checking and Q&A workflows in parallel
    """
    paper_chunks = read_and_chunk_file(paper_filepath)
    combined_state = CombinedPaperState()
    fact_checker = create_fact_checker_graph()
    qa_graph = create_graph(paper_filepath)
    fact_check_tasks = []
    for chunk in paper_chunks:
        task = asyncio.create_task(fact_checker.invoke({"inputs": chunk}))
        fact_check_tasks.append(task)
    qa_task = asyncio.create_task(
        qa_graph.ainvoke({"paper": {"filepath": paper_filepath}, "messages": []},
                         {"recursion_limit": 10000})
    )

    fact_check_results = await asyncio.gather(*fact_check_tasks)
    qa_results = await qa_task
    overall_factual_score = 0
    total_claims = 0
    all_fact_pairs = []

    for result in fact_check_results:
        all_fact_pairs.extend(result["pairs"])
        overall_factual_score += result["total_score"]
        total_claims += result["no_claims"]

    average_factual_score = (
        overall_factual_score / total_claims if total_claims > 0 else 0
    )
    is_factual = average_factual_score > 3
    combined_state.fact_checker_results = {
        "average_score": average_factual_score,
        "is_factual": is_factual,
        "pairs": all_fact_pairs,
        "total_claims": total_claims,
    }

    combined_state.qa_results = {
        "publishability": qa_results.publishability,
        "suggestions": qa_results.suggestions,
        "queries": qa_results.queries,
    }

    combined_state.overall_assessment = generate_overall_assessment(
        is_factual,
        average_factual_score,
        qa_results.publishability,
        qa_results.suggestions,
    )

    combined_state.is_reliable = is_factual and qa_results.publishability == "Publish"

    return combined_state


def generate_overall_assessment(is_factual, factual_score, publishability, suggestions):
    """Generate an overall assessment based on both fact checking and Q&A results"""
    if is_factual and publishability == "Publish":
        return f"This paper is factual (score: {factual_score:.2f}/5) and recommended for publication."

    issues = []

    if not is_factual:
        issues.append(
            f"The paper contains factual inaccuracies (factuality score: {factual_score:.2f}/5)"
        )

    if publishability != "Publish":
        issues.append(f"The paper has concerns regarding publishability: {suggestions}")

    return "This paper has the following issues:\n- " + "\n- ".join(issues)


if __name__ == "__main__":
    paper_filepath = "/home/naba/Desktop/PRAGATI/Tiny _ML_Things.pdf"
    results = asyncio.run(process_paper_parallel(paper_filepath))
    print(results)
