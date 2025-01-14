import os
from PIL import Image
import io
import pprint
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utils.schemas import Reviewer, Queries, QAPair, Paper
from utils.helper import print_reviewers

from agentstates import QuestionState, AnswerState, IntermediateAnswerState
from ask_questions import create_researchers, get_questionnaire
from compile_questions import compile_questions, compile2
from answer_questions import generate_sub_queries, retrieve_references, answer_sub_queries, summarise_results
from fact_check import *

from dotenv import load_dotenv

load_dotenv()

paper = Paper(object_id=os.getenv("TEMP_PAPER_OBJ_ID"), title="Sample Paper", filename="sample_paper.pdf")
initial_state_dict = {
    "messages": [],
    "paper": paper,
    "num_reviewers": 3,
    "reviewers": [],
    "conference": "NeurIPS",
    "conference_description": "The NeurIPS (Conference on Neural Information Processing Systems) is a leading annual conference in machine learning and artificial intelligence, featuring cutting-edge research in neural computation, deep learning, and related fields. It attracts researchers, practitioners, and industry leaders from around the world."
}
initial_state = QuestionState(**initial_state_dict)

def are_facts_correct(state: FactCheckerState):
    if state['Decision']:
        return 'ask_questions'
    else:
        return '__end__'

# main_graph_builder = StateGraph(input=QuestionState, output=AnswerState)
# main_graph_builder.add_node("create_researchers", create_researchers)
# main_graph_builder.add_node("get_questionnaire", get_questionnaire)
# main_graph_builder.add_node("compile_questions", compile_questions)
# main_graph_builder.add_node("generate_sub_queries", generate_sub_queries)
# main_graph_builder.add_node("retrieve_references", retrieve_references)
# main_graph_builder.add_node("answer_sub_queries", answer_sub_queries)
# main_graph_builder.add_node("summarise_results", summarise_results)

# main_graph_builder.add_edge(START, "create_researchers")
# main_graph_builder.add_edge("create_researchers", "get_questionnaire")
# main_graph_builder.add_edge("get_questionnaire", "compile_questions")
# main_graph_builder.add_edge("compile_questions", "generate_sub_queries")
# main_graph_builder.add_edge("generate_sub_queries", "retrieve_references")
# main_graph_builder.add_edge("retrieve_references", "answer_sub_queries")
# main_graph_builder.add_edge("answer_sub_queries", "summarise_results")
# main_graph_builder.add_edge("summarise_results", END)
builder = StateGraph(Generatecheckertate)
builder.add_node("create_checker", create_checker)
builder.add_edge(START, "create_checker")
builder.add_edge("create_checker",END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

question_prompt = """you are checking facts for the Fact checker:{goals}
                    \n\n and the facts that you are going to check is as following 
                    \n\n\n {facts}"""
search_prompt = SystemMessage(content=f"""You are going to fact check the given facts as an offcial fact checker for a conference""")

answer_prompt = """you are:\n{goals}\n now you have to generate an answer for the given fact
 based on your searches which is as following:\n\n {context}
"""

reason_prompt = """Given the data in the document as following\n\n{fact} \n\nfind the reason for its correctness using\n\n {context}"""

web_search_builder = StateGraph(CheckingState)
web_search_builder.add_node("search_query", generate_question)
web_search_builder.add_node("web_search",search_web)
web_search_builder.add_node("Arxiv_search",search_arxiv)
web_search_builder.add_node("Gscholar_search",search_gscholar)
web_search_builder.add_node("fact_correctness",generate_answer)
web_search_builder.add_node("save_search",save_search)
web_search_builder.add_node("reason_writer",state_reason)
#Logic of Web Search
web_search_builder.add_edge(START,"search_query")
web_search_builder.add_edge("search_query","web_search")
web_search_builder.add_conditional_edges("search_query",route_websearch,["Arxiv_search","Gscholar_search"])
web_search_builder.add_conditional_edges("Arxiv_search",route_search,["fact_correctness"])
web_search_builder.add_conditional_edges("Gscholar_search",route_search,["fact_correctness"])
web_search_builder.add_edge("web_search","fact_correctness")
web_search_builder.add_conditional_edges("fact_correctness",route_messages,["search_query","save_search"])
web_search_builder.add_edge("save_search","reason_writer")
web_search_builder.add_edge("reason_writer",END)

memory = MemorySaver()
web_search_graph = web_search_builder.compile(checkpointer=memory).with_config(run_name="Web search")

sanity_prompt = """given the context in the document as following:\n\n{facts}
                \n\n and now verify this based on {reason} 
                """

builder = StateGraph(FactCheckerState)
builder.add_node("create_checkers", create_checker)
builder.add_node("store_facts",web_search_builder.compile())
builder.add_node("correctness",state_reason)
builder.add_node("sanity",Decide_sanity)
builder.add_node("scoring",Score_document)

builder.add_edge(START, "create_checkers")
builder.add_edge("create_checkers", "store_facts")
builder.add_edge("store_facts", "correctness")
builder.add_edge("correctness", "sanity")
builder.add_edge("sanity", "scoring")
builder.add_edge("scoring", END)


memory = MemorySaver()
Checker_graph = builder.compile(checkpointer=memory).with_config(run_name="Fact Checker")

ask_questions_builder = StateGraph(QuestionState)
memory = MemorySaver()
ask_questions_builder.add_node("create_researchers", create_researchers)
ask_questions_builder.add_node("get_questionnaire", get_questionnaire)

ask_questions_builder.add_edge(START, "create_researchers")
ask_questions_builder.add_edge("create_researchers", "get_questionnaire")
ask_questions_builder.add_edge("get_questionnaire", END)

answer_questions_builder = StateGraph(AnswerState)

answer_questions_builder.add_node("generate_sub_queries", generate_sub_queries)
answer_questions_builder.add_node("retrieve_references", retrieve_references)
answer_questions_builder.add_node("answer_sub_queries", answer_sub_queries)
answer_questions_builder.add_node("summarise_results", summarise_results)

answer_questions_builder.add_edge(START, "generate_sub_queries")
answer_questions_builder.add_edge("generate_sub_queries", "retrieve_references")
answer_questions_builder.add_edge("retrieve_references", "answer_sub_queries")
answer_questions_builder.add_edge("answer_sub_queries", "summarise_results")
answer_questions_builder.add_edge("summarise_results", END)

# compile_questions_builder = StateGraph(input=QuestionState, output=AnswerState)
# compile_questions_builder.add_node("compile_questions", compile_questions)
# # compile_questions_builder.add_node("compile2", compile2)

# compile_questions_builder.add_edge(START, "compile_questions")
# compile_questions_builder.add_edge("compile_questions", "compile2")
# compile_questions_builder.add_edge("compile2", END)

# compile_questions_graph = compile_questions_builder.compile()

main_graph_builder = StateGraph(input=QuestionState, output=AnswerState)

main_graph_builder.add_node("fact_checker", Checker_graph)
main_graph_builder.add_node("ask_questions", ask_questions_builder.compile())
main_graph_builder.add_node("compile_questions", compile_questions)
# main_graph_builder.add_node("compile2", compile2)
main_graph_builder.add_node("answer_questions", answer_questions_builder.compile())

# main_graph_builder.add_edge(START, "ask_questions")
# main_graph_builder.add_edge("ask_questions", "compile_questions")
# main_graph_builder.add_edge("compile_questions", "compile2")
# main_graph_builder.add_edge("compile2", "answer_questions")
# main_graph_builder.add_edge("answer_questions", END)
main_graph_builder.add_edge(START, "fact_checker")
main_graph_builder.add_conditional_edges("fact_checker", are_facts_correct, ["ask_questions", "__end__"])
main_graph_builder.add_edge("ask_questions", "compile_questions")
main_graph_builder.add_edge("compile_questions", "answer_questions")
main_graph_builder.add_edge("answer_questions", END)

main_graph = main_graph_builder.compile()

thread = {
            "configurable": 
                {
                    "thread_id": os.getenv("TEMP_PAPER_OBJ_ID")
                },
         }

image_bytes = main_graph.get_graph(xray=2).draw_mermaid_png(scale=2)
image = PILImage.open(io.BytesIO(image_bytes))
image.save('final_graph.png')

# for event in main_graph.stream(initial_state, thread):
#     # print(event)
#     for key, value in event.items():
#         print("Output from node: ", end=' ')
#         pprint.pprint(key)
#         pprint.pprint("-------------------------")
#         pprint.pprint(value)
#         final_state = value
#     time.sleep(2)

pprint.pprint("Final State:")
# pprint.pprint(final_state)