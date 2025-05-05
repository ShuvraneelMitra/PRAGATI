from agents.answer import agen_graph
from agents.persona import qgen_graph
from agents.schemas import TokenTracker, Paper
from agents.states import QuestionState
from langgraph.graph import StateGraph, END, START
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_combined_graph():
    """
    Creates a LangGraph where:
    - Node 1: Generates questions (qgen_graph)
    - Node 2: Answers the questions (agen_graph)
    - Then ends
    """
    builder = StateGraph(QuestionState)

    builder.add_node("generate_questions", qgen_graph())
    builder.add_node("generate_answers", agen_graph())

    builder.set_entry_point("generate_questions")
    builder.add_edge("generate_questions", "generate_answers")
    builder.set_finish_point("generate_answers")

    graph = builder.compile()
    return graph


if __name__ == "__main__":
    paper = Paper(
        filepath="/home/naba/Desktop/PRAGATI/mrinmoy.pdf",
        title="Do we really need Foundation Models for multi-step-ahead Epidemic Forecasting?",
        topic="ML and Time Series",
        sections=["Introduction", "Methodology", "Conclusion"],
    )

    qstate = QuestionState(
        messages=[],
        paper=paper,
        num_reviewers=1,
        token_usage=TokenTracker(
            net_input_tokens=0,
            net_output_tokens=0,
            net_tokens=0
        ),
        reviewers=[],
        queries=[],
    )
    graph = run_combined_graph()
    final_state = graph.invoke(qstate)
    final_state = QuestionState(**final_state)
    print("\nQA Pipeline Results:")
    print(f"Publishability: {final_state.publishability}")
    print(f"Suggestions: {final_state.suggestions}")
    print(f"Q&A Pairs: {len(final_state.queries)}")
