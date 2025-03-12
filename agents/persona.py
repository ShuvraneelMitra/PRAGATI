from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from states import QuestionState
from schemas import Reviewer

import os
import yaml
from dotenv import load_dotenv
from typing import List
load_dotenv()

with open('../utils/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)


########################################################################################################################
def generate_reviewers(num_reviewers: int) -> List[Reviewer]:
    """
    Generates `num_reviewers` reviewers to simulate a real-life review
    scenario. Keep in mind that usually the list of reviewers
    for a paper includes both junior and senior reviewers and
    also sometimes people working in an entirely different domain
    than the paper's topic.

    Args:
        state: the state object
        num_reviewers: an integer describing the total number of reviewers
    Returns: A list of probabilistically generated list of reviewers
    """
    list_reviewers: List[Reviewer] = []
    for i in range(num_reviewers):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.9,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY")
        )

        messages = [
            ("system", prompts["QuestionAnswering"]["generate_reviewers"]["system"]),
            ("human", prompts["QuestionAnswering"]["generate_reviewers"]["human"]),
        ]
        reviewer = llm.with_structured_output(Reviewer,
                                              method="json_mode",
                                              include_raw=False).invoke(messages)
        list_reviewers.append(reviewer)

    return list_reviewers

# graph_builder = StateGraph()
# graph_builder.compile().get_graph().draw_mermaid_png()

print(generate_reviewers(4))
