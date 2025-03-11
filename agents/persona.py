from typing import Sequence
from langgraph.graph import StateGraph, START, END
from schemas import Reviewer
from states import QuestionState

from dotenv import load_dotenv

load_dotenv()


def generate_reviewers(state: QuestionState) -> QuestionState:
    """
    Generates 4 reviewers to simulate a real-life review
    scenario. Keep in mind that usually the list of reviewers
    for a paper includes both junior and senior reviewers and
    also sometimes people working in an entirely different domain
    than the paper's topic.

    Returns: A list of probabilistically generated list of reviewers
    """



graph_builder = StateGraph()
graph_builder.compile().get_graph().draw_mermaid_png()
