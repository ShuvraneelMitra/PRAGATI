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
from langgraph.graph import StateGraph, START, END
import os
import yaml
import json

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
                qa_pair.answer = response["result"].lower().strip() == "yes"

                state.token_usage.net_input_tokens += response["input_tokens"]
                state.token_usage.net_output_tokens += response["output_tokens"]
                state.token_usage.net_tokens = (
                    state.token_usage.net_output_tokens
                    + state.token_usage.net_input_tokens
                )

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
                            \n {single_query.query} : {single_query.answer} \n
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
                    -prompts["QuestionAnswering"]["review_and_suggest"]["human"],
                    {
                        "queries": f"{[(query.question, query.answer) for query in state.queries]}"
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
        print(
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
