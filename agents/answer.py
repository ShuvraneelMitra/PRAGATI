import gradio.utils

from agents.states import QuestionState
from pdfparse.rag_llama import RAG
from utils.utils import tmpl_to_prompt
import logging
import gradio
from typing import Annotated, Literal, Sequence
from llama_index.core.retrievers import BaseRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph.state import CompiledStateGraph
from agents.schemas import TokenTracker
from langgraph.graph import StateGraph, START, END
import os
import yaml
import json

from agents.schemas import Paper, QAPair
from agents.states import SingleQuery

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.prebuilt import tools_condition

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="PRAGATI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

with open("utils/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

########################################################################################################################


def answerer(state: QuestionState) -> QuestionState:
    logger.info(f"Starting RAG application with PDF: {state.paper.filepath}")
    rag = RAG(state.paper.filepath)
    retriever = rag.create_retriever(rag.create_db())

    for reviewer in state.queries:
        for single_query in reviewer:
            for qa_pair in single_query.sub_queries:
                query_text = f"""
                You are given a query about the context document. You need to answer it in one word, 
                either a YES or a NO, without generating any extra verbiage whatsoever.
                
                The query is: {qa_pair.query}
                """
                logger.info(f"User query: {query_text}")
                response = rag.rag_query(query_text, retriever)
                logger.info("======================Debug======================")
                logger.info(f"query: {qa_pair.query}")
                logger.info(f"response: {response['result']}")
                qa_pair.answer = response["result"].lower().strip() == "yes"
                if state.token_usage is None:
                    state.token_usage = TokenTracker(net_input_tokens=0, net_output_tokens=0, net_tokens=0)

                state.token_usage.net_input_tokens += response["input_tokens"]
                state.token_usage.net_output_tokens += response["output_tokens"]
                state.token_usage.net_tokens = (
                    state.token_usage.net_output_tokens
                    + state.token_usage.net_input_tokens
                )
                logger.info(f"query: {qa_pair.query}")
                logger.info(f"Answer: {qa_pair.answer}")
                logger.info(f"Answer: {qa_pair.answer}")
                
    return dict(queries=state.queries, token_usage=state.token_usage)


def compiler(state: QuestionState) -> QuestionState:
    logger.info(f"Starting RAG application with PDF: {state.paper.filepath}")
    rag = RAG(state.paper.filepath)
    retriever = rag.create_retriever(rag.create_db())

    for reviewer in state.queries:
        for single_query in reviewer:
            compiled = str()
            for qa_pair in single_query.sub_queries:
                compiled += f"""
                            \n {single_query.question} : {single_query.answer} \n
                            """

            query_text = f"""
                You are given a main question about the document and multiple subqueries along with their answers.
                Based on the subqueries and their answers, along with the document, generate a concise but thorough 
                answer to the main question.
    
                The main question is: {single_query.question} 
                
                and the subqueries are:
                
                f{compiled}
                """
            logger.info(f"User query: {single_query.question}")
            response = rag.rag_query(query_text, retriever)
            single_query.answer = response["result"]
            logger.info("======================Debug======================")
            logger.info(f"Answer: {single_query.answer}")
            state.token_usage.net_input_tokens += response["input_tokens"]
            state.token_usage.net_output_tokens += response["output_tokens"]
            state.token_usage.net_tokens = (
                state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
            )

    return dict(queries=state.queries, token_usage=state.token_usage)


def review_and_suggest(state: QuestionState) -> QuestionState:
    for i in range(len(state.queries)):
        messages = [
            (
                "system",
                prompts["QuestionAnswering"]["review_and_suggest"]["system"],
            ),
            (
                "human",
                tmpl_to_prompt(
                    prompts["QuestionAnswering"]["review_and_suggest"]["human"],
                    {
                        "queries": f"{[(query[i].question, query[i].answer) for query in state.queries]}"
                    },
                ),
            ),
        ]

        response = llm.invoke(messages)
        logger.info("======================Debug======================")
        logger.info(f"response: {response.content}")
        try:
            subq_json = json.loads(response.content)
        except json.decoder.JSONDecodeError as json_error:
            logger.info(
                f"JSON Decode Error: {json_error}\nThrown because the LLM output is {response.content!r}"
            )
            raise RuntimeError("Could not parse LLM output as JSON") from json_error

        if not isinstance(subq_json, dict):
            raise TypeError(f"Expected JSON object (dict), got {type(subq_json).__name__}: {subq_json!r}")

        required = ("publishability", "suggestions")
        missing = [k for k in required if k not in subq_json]
        if missing:
            raise KeyError(f"LLM response is missing keys {missing!r} in {subq_json!r}")
        
        if len(state.reviewers) >0:
            state.reviewers[i].review = subq_json["publishability"]
            state.reviewers[i].suggestions = subq_json["suggestions"]


        state.token_usage.net_input_tokens += response.response_metadata["token_usage"][
            "prompt_tokens"
        ]
        state.token_usage.net_output_tokens += response.response_metadata[
            "token_usage"
        ]["completion_tokens"]
        state.token_usage.net_tokens = (
            state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
        )

    logger.info("======================Debug======================")
    logger.info(f"publishability: {[reviewer.review for reviewer in state.reviewers]}")
    logger.info(f"suggestions: {[reviewer.suggestions for reviewer in state.reviewers]}")
    logger.info(f"token_usage: {state.token_usage}")
    return dict(reviewers=state.reviewers, token_usage=state.token_usage)


def summary(state: QuestionState) -> QuestionState:
    messages = [
        (
            "system",
            prompts["QuestionAnswering"]["summary"]["system"],
        ),
        (
            "human",
            tmpl_to_prompt(
                prompts["QuestionAnswering"]["summary"]["human"],
                {
                    "reviews": f"{[reviewer.review for reviewer in state.reviewers]}",
                    "suggestions": f"{[reviewer.suggestions for reviewer in state.reviewers]}",
                },
            ),
        ),
    ]

    response = llm.invoke(messages)
    try:
        subq_json = json.loads(response.content)
    except json.decoder.JSONDecodeError as json_error:
        logger.info(
            f"JSON Decode Error: {json_error}\n Thrown because the LLM output is {response.content}"
        )
        return dict()

    state.publishability = subq_json["publishability"]
    state.suggestions = subq_json["suggestions"]

    state.token_usage.net_input_tokens += response.response_metadata["token_usage"][
        "prompt_tokens"
    ]
    state.token_usage.net_output_tokens += response.response_metadata["token_usage"][
        "completion_tokens"
    ]
    state.token_usage.net_tokens = (
        state.token_usage.net_output_tokens + state.token_usage.net_input_tokens
    )
    logger.info("======================Debug======================")
    logger.info(f"publishability: {state.publishability}")
    logger.info(f"suggestions: {state.suggestions}")
    logger.info(f"token_usage: {state.token_usage}")
    return dict(
        publishability=state.publishability,
        suggestions=state.suggestions,
        token_usage=state.token_usage,
    )


def agen_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(QuestionState)
    graph_builder.add_node("answerer", answerer)
    graph_builder.add_node("compiler", compiler)
    graph_builder.add_node("review_and_suggest", review_and_suggest)
    graph_builder.add_edge(START, "answerer")
    graph_builder.add_edge("answerer", "compiler")
    graph_builder.add_edge("compiler", "review_and_suggest")

    graph = graph_builder.compile()
    return graph

if __name__ == "__main__":
    graph = agen_graph()
    state = QuestionState(
    paper=Paper(
        filepath="/home/naba/Desktop/PRAGATI/satya.pdf",
        title="Some Title",
        topic="ML",
        filename="satya.pdf",
        sections=["Abstract", "Introduction", "Methodology"]
    ),
    queries=[
        [
            SingleQuery(
                question="Is the methodology novel?",
                answer="Yes",
                sub_queries=[
                    QAPair(query="Does it introduce a new technique?", answer=True),
                    QAPair(query="Is the implementation clear?", answer=False),
                    QAPair(query="Has it been validated experimentally?", answer=True)
                ]
            )
        ]
    ]
)
    result = graph.invoke(state)
    logger.info(result)