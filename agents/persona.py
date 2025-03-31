from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_groq import ChatGroq
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

from agents.states import QuestionState, SingleQuery
from agents.schemas import Reviewer, TokenTracker, Paper, QAPair
from utils.utils import tmpl_to_prompt

import os
import yaml
import json
from dotenv import load_dotenv
from pprint import pprint
from typing import List
from PIL import Image
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

with open("utils/prompts.yaml", "r") as file:
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
def generate_reviewers(state: QuestionState) -> QuestionState:
    """
    Generates reviewers to simulate a real-life review
    scenario. Keep in mind that usually the list of reviewers
    for a paper includes both junior and senior reviewers and
    also sometimes people working in an entirely different domain
    than the paper's topic.

    Args:
        state: the state object

    Returns: A list of reviewers which follow the Reviewer schema
    """

    list_reviewers: List[Reviewer] = []
    ai_messages: list[BaseMessage] = []

    for i in range(state.num_reviewers):
        messages = [
            ("system", prompts["QuestionAnswering"]["generate_reviewers"]["system"]),
            (
                "human",
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["generate_reviewers"]["human"],
                    {"topics": state.paper.topic},
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
        num_reviewers=state.num_reviewers,
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

    state.queries = [
        [SingleQuery() for _ in range(len(state.paper.sections))]
        for _ in range(state.num_reviewers)
    ]
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


def generate_subqueries(state: QuestionState) -> QuestionState:
    """
    Generates subqueries for each question asked by the reviewers
    Args:
        state: the state object
        num_subqueries: the number of subqueries to be generated per broad question
    Returns: a set of subqueries
    """
    ai_messages = []
    for i, reviewer in enumerate(state.queries):
        for j, single_query in enumerate(reviewer):

            messages = [
                (
                    "system",
                    prompts["QuestionAnswering"]["generate_subqueries"]["system"],
                ),
                (
                    "human",
                    tmpl_to_prompt(
                        prompts["QuestionAnswering"]["generate_subqueries"]["human"],
                        {
                            "topics": state.paper.topic,
                            "num_subqueries": f"{state.num_subqueries}",
                            "query": single_query.question,
                            "specialisation": state.reviewers[i].specialisation,
                        },
                    ),
                ),
            ]

            response = llm.invoke(messages)

            try:
                subq_json = json.loads(response.content)
            except json.decoder.JSONDecodeError as json_error:
                print(
                    f"JSON Decode Error: {json_error}\n Thrown because the LLM output is {response.content}"
                )
                return dict()

            state.queries[i][j].sub_queries = [
                QAPair(query=q) for q in subq_json["sub-queries"]
            ]
            ai_messages.append(response)

            state.token_usage.net_input_tokens += response.response_metadata[
                "token_usage"
            ]["prompt_tokens"]
            state.token_usage.net_output_tokens += response.response_metadata[
                "token_usage"
            ]["completion_tokens"]
            state.token_usage.net_tokens = (
                state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
            )

    messages += ai_messages

    return dict(messages=messages, token_usage=state.token_usage, queries=state.queries)


def qgen_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(QuestionState)
    graph_builder.add_node("generate_reviewers", generate_reviewers)
    graph_builder.add_node("generate_questions", generate_questions)
    graph_builder.add_node("generate_subqueries", generate_subqueries)
    graph_builder.add_edge(START, "generate_reviewers")
    graph_builder.add_edge("generate_reviewers", "generate_questions")
    graph_builder.add_edge("generate_questions", "generate_subqueries")

    graph = graph_builder.compile()
    return graph


if __name__ == "__main__":
    graph = qgen_graph()
    graph_image = graph.get_graph().draw_mermaid_png(
        curve_style=CurveStyle.LINEAR,
        node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
        wrap_label_n_words=9,
        output_file_path=None,
        draw_method=MermaidDrawMethod.PYPPETEER,
        background_color="white",
        padding=10,
    )

    if not os.path.exists("../assets"):
        os.makedirs("../assets")
    with open("../assets/generate_subqueries.png", "wb") as f:
        f.write(graph_image)

    show_img = input("Show generated graph? Y/n ")
    if show_img.lower().strip() == "y":
        img = Image.open("../assets/generate_subqueries.png")
        img.show()

    ex = {
        "messages": [],
        "paper": Paper(
            topic="Large Language Models",
            sections=["Introduction", "Methodology", "Conclusion"],
        ),
        "num_reviewers": 4,
        "token_usage": TokenTracker(
            net_input_tokens=0, net_output_tokens=0, net_tokens=0
        ),
        "reviewers": [],
        "queries": [],
    }

    final_state = graph.invoke(QuestionState(**ex))
    pprint(final_state)
