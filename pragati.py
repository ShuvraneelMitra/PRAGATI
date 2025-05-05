from typing import Dict, List, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import yaml
import logging
from agents.schemas import Paper
from agents.states import PaperState, QuestionState, FactCheckerState, TokenTracker
from agents.critic import run_combined_graph
from agents.checker import create_fact_checker_graph
import os
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

with open("utils/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)


def qa_node(state: PaperState) -> dict:
    critic_graph = run_combined_graph()
    updated_qa_results = critic_graph.invoke(state.qa_results)
    return {"qa_results": updated_qa_results}


def fact_checker_node(state: PaperState) -> dict:
    fact_checker_graph = create_fact_checker_graph()
    updated_fc_state = fact_checker_graph.invoke(
        state.fact_checker_results,
        {"recursion_limit": 1000}
    )
    return {"fact_checker_results": updated_fc_state}

def combine_results_node(state: PaperState) -> PaperState:
    state.overall_assesment.factual = state.fact_checker_results.is_factual
    state.overall_assesment.fact_checker_score = state.fact_checker_results.average_score
    state.overall_assesment.Publishability = state.qa_results.publishability
    state.overall_assesment.Suggestions = state.qa_results.suggestions
    return state


def PRAGATI_pipeline():
    builder = StateGraph(PaperState)
    builder.add_node("qa_node", qa_node)
    builder.add_node("fact_checker_node", fact_checker_node)
    builder.add_node("combine_results_node", combine_results_node)

    builder.add_edge(START, "qa_node")
    builder.add_edge(START, "fact_checker_node")

    builder.add_edge(["qa_node", "fact_checker_node"], "combine_results_node")
    builder.add_edge("combine_results_node", END)

    compiled_graph = builder.compile()

    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pragati_graph.png")
        graph_image = compiled_graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
        with open(output_path, "wb") as f:
            f.write(graph_image)
        logger.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")

    return compiled_graph


if __name__ == "__main__":
    paper = Paper(
        filepath="/home/naba/Desktop/PRAGATI/mrinmoy.pdf",
        title="Do we really need Foundation Models for multi-step-ahead Epidemic Forecasting?",
        topic="ML and Time Series",
        sections=["Introduction", "Methodology", "Conclusion"],
    )

    qa_input = QuestionState(
        messages=[],
        paper=paper,
        num_reviewers=1,
        token_usage=TokenTracker(
            net_input_tokens=0, net_output_tokens=0, net_tokens=0
        ),
        reviewers=[],
        queries=[],
    )

    fc_input = FactCheckerState(paper=paper)

    state = PaperState(
        qa_results=qa_input,
        fact_checker_results=fc_input
    )

    pragati_graph = PRAGATI_pipeline()
    final_state = pragati_graph.invoke(state, {"recursion_limit": 1000})
    

    paper_state = PaperState(**final_state)

    logger.info("Final Overall Assessment:")
    logger.info(f"Factual: {paper_state.overall_assesment.factual}")
    logger.info(f"Fact Checker Score: {paper_state.overall_assesment.fact_checker_score}")
    logger.info(f"Publishability: {state.qa_results.publishability}")
    logger.info(f"Suggestions: {state.qa_results.suggestions}")
    
    print("\nFinal Overall Assessment:\n")
    print(f"Factual: {paper_state.overall_assesment.factual}")
    print(f"Fact Checker Score: {paper_state.overall_assesment.fact_checker_score}")
    print(f"Publishability: {state.qa_results.publishability}")
    print(f"Suggestions: {state.qa_results.suggestions}")
