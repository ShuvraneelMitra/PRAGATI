from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from agents.states import QuestionState
from agents.schemas import Reviewer, TokenTracker, Paper
from utils.utils import tmpl_to_prompt

import os
import yaml
import json
from dotenv import load_dotenv
from pprint import pprint
from typing import List

load_dotenv()

with open("../utils/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)


########################################################################################################################
def generate_reviewers(state: QuestionState, num_reviewers: int) -> QuestionState:
    """
    Generates `num_reviewers` reviewers to simulate a real-life review
    scenario. Keep in mind that usually the list of reviewers
    for a paper includes both junior and senior reviewers and
    also sometimes people working in an entirely different domain
    than the paper's topic.

    Args:
        state: the state object
        num_reviewers: an integer describing the total number of reviewers
    Returns: A list of reviewers which follow the Reviewer schema
    """

    list_reviewers: List[Reviewer] = []
    ai_messages: list[BaseMessage] = []

    for i in range(num_reviewers):
        messages = [
            ("system", prompts["QuestionAnswering"]["generate_reviewers"]["system"]),
            (
                "human",
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["generate_reviewers"]["human"],
                    {"topics": state.topic},
                ),
            ),
        ]
        response = llm.invoke(messages)

        ai_messages.append(response)
        reviewer = response.content.replace("\n", "")
        if isinstance(reviewer, dict):
            list_reviewers.append(Reviewer(**reviewer))

        elif isinstance(reviewer, str):
            try:
                json_data = json.loads(reviewer)
                if isinstance(json_data, dict):
                    list_reviewers.append(Reviewer(**json_data))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string {json.loads(reviewer)}")

        state.token_usage.net_input_tokens += response.response_metadata["token_usage"][
            "prompt_tokens"
        ]
        state.token_usage.net_output_tokens += response.response_metadata[
            "token_usage"
        ]["completion_tokens"]
        state.token_usage.net_tokens = (
            state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
        )

    return dict(
        messages=[
            SystemMessage(prompts["QuestionAnswering"]["generate_reviewers"]["system"]),
            HumanMessage(
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["generate_reviewers"]["human"],
                    {"topics": state.paper.topic},
                )
            ),
            *ai_messages,
        ],
        reviewers=list_reviewers,
        num_reviewers=num_reviewers,
        token_usage=state.token_usage,
    )


def generate_questions(state: QuestionState) -> QuestionState:
    """
    Generates some questions about each section of the paper
    Args:
        state: the state object

    Returns: A list of questions per reviewer on each of the sections
    """
    messages = (
        [("system", prompts["QuestionAnswering"]["generate_questions"]["system"])]
        + state.messages[-state.num_reviewers :]
        + [
            (
                "human",
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["generate_questions"]["human"],
                    {
                        "topic": state.paper.topic,
                        "specialisations": f"{[reviewer.specialisation for reviewer in state.reviewers]}",
                        "num_reviewers": f"{state.num_reviewers}",
                        "sections": f"{state.paper.sections}",
                    },
                ),
            ),
        ]
    )

    response = llm.invoke(messages)
    state.token_usage.net_input_tokens += response.response_metadata["token_usage"][
        "prompt_tokens"
    ]
    state.token_usage.net_output_tokens += response.response_metadata["token_usage"][
        "completion_tokens"
    ]
    state.token_usage.net_tokens = (
        state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
    )

    list_questions = []
    json_questions = json.loads(response.content)
    for reviewer in json_questions.keys():
        list_questions.append(list(json_questions[reviewer]["questions"].values()))

    state.queries = [[SingleQuery() for _ in len(state.paper.sections)] for _ in state.num_reviewers]
    for i, reviewer in enumerate(state.queries):
        for j, single_query in enumerate(reviewer):
            single_query.question = list_questions[i][j]

    return dict(
        messages=[
            (
                "human",
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["generate_questions"]["human"],
                    {
                        "topic": state.paper.topic,
                        "specialisations": f"{[reviewer.specialisation for reviewer in state.reviewers]}",
                        "num_reviewers": f"{state.num_reviewers}",
                        "sections": f"{state.paper.sections}",
                    },
                ),
            )
        ]
        + [response],
        token_usage=state.token_usage,
        queries=state.queries,
    )


def generate_subqueries(state: QuestionState, num_subqueries: int) -> QuestionState:
    """
    Generates subqueries for each question asked by the reviewers
    Args:
        state: the state object
        num_subqueries: the number of subqueries to be generated per broad question
    Returns: a set of subqueries
    """
    pass


# graph_builder = StateGraph()
# graph_builder.compile().get_graph().draw_mermaid_png()
