
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
# from parser.parser import PDFTextExtractor
import pandas as pd
from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

from typing import Annotated
from langgraph.graph import MessagesState
from decimal import Decimal
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import get_buffer_string
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import ArxivAPIWrapper
from langgraph.constants import Send
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper

from agentstates import Factchecker, Perspectives, GenerateCheckerState, CheckingState, SearchQuery, FactCheckerState

load_dotenv()

def make_lm(model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=5000):
    """
    Invoke the LLM with the given messages and handle response safely.
    """
    llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)    
    return llm

llm = make_lm()

system_instructions = """
                    You are tasked with creating a set of AI fact checker personas. Follow these instructions carefully:

                    1. First, review the research topic:
                    {topic}

                    2. Determine the most interesting themes based upon documents and / or feedback above.
                                        
                    3. Pick the top {max_checker} themes.

                    4. Assign one checker to each theme.
                    """

def create_checker(state: GenerateCheckerState):
    """ Create checker """   
    topic = state['topic']
    max_checker = state['max_checker']
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = system_instructions.format(topic=topic, max_checker=max_checker)
    checker = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of checker.")])
    state["checker"] = checker.checker 
    return state

builder = StateGraph(GenerateCheckerState)
builder.add_node("create_checker", create_checker)
builder.add_edge(START, "create_checker")
builder.add_edge("create_checker",END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
image_bytes = graph.get_graph(xray=1).draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_bytes))
image.save('graph_output.png')

thread = {"configurable": {"thread_id": "1"}}

question_prompt = """you are checking facts for the Fact checker:{goals}
                    \n\n and the facts that you are going to check is as following 
                    \n\n\n {facts}"""
search_prompt = SystemMessage(content=f"""You are going to fact check the given facts as an offcial fact checker for a conference""")

def route_websearch(state: CheckingState):
    checker = state["fchecker"]
    web = checker.web
    if "https://arxiv.org/" in web:
        return "Arxiv_search"
    
    elif "https://scholar.google.com/" in web:
        return "Gscholar_search"
    
def route_search(state: CheckingState):
    checker = state["fchecker"]
    web = checker.web()
    if "https://arxiv.org/" in web:
        return "fact_correctness"
    elif "https://scholar.google.com/" in web:
        return "fact_correctness"

def generate_question(state: CheckingState):
    """ Search query generation for the websearch"""
    checker = state["fchecker"]
    messages = state["messages"]

    system_message = question_prompt.format(goals=checker.persona,facts=state["fact"])
    question = llm.invoke([SystemMessage(content=system_message)]+messages)
    print(question.content)
    state["messages"] = [question]
    return state

tavily_search = TavilySearchResults(max_results=3)

def search_web(state: CheckingState):
    
    """ Retrieve docs from web search """
    
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_prompt]+state['messages'])
    search_docs = tavily_search.invoke(search_query.search_query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    print(formatted_search_docs)
    state["context"] = [formatted_search_docs]

    return state 

def search_arxiv(state: CheckingState):
    structured_llm = llm.with_structured_output(SearchQuery)

    # Define search prompt template
    search_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate a concise search query for finding academic papers about {topic} on arXiv."
    )
    search_query = structured_llm.invoke(search_prompt.format_prompt(topic=topic).to_string())
    arxiv_search = ArxivAPIWrapper()
    search_docs = arxiv_search.run(search_query.search_query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["entry_id"]}">\nTitle: {doc["title"]}\nSummary: {doc["summary"]}\n</Document>'
            for doc in search_docs.get("docs", [])  # Adapt this based on API wrapper output structure
        ]
    )
    state["context"] = [formatted_search_docs]
    return state    

def search_gscholar(state: CheckingState):
    structured_llm = llm.with_structured_output(SearchQuery)
    # Initialize the Google Scholar API wrapper
    gscholar_search = GoogleScholarAPIWrapper()
    search_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate a concise search query for finding academic papers about {topic} on Google Scholar."
    )
    search_query = structured_llm.invoke(search_prompt.format_prompt(topic=topic).to_string())
    search_docs = gscholar_search.run(search_query.search_query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["link"]}">\nTitle: {doc["title"]}\nAuthors: {", ".join(doc["authors"])}\nSummary: {doc["snippet"]}\n</Document>'
            for doc in search_docs.get("docs", [])  # Adapt this based on API wrapper output structure
        ]
    )
    state["context"] = [formatted_search_docs]
    return state


answer_prompt = """you are:\n{goals}\n now you have to generate an answer for the given fact
 based on your searches which is as following:\n\n {context}
"""

def generate_answer(state: CheckingState):
    """ Node to answer a question """
    checker = state["fchecker"]
    messages = state["messages"]
    context = state["context"]
    system_message = answer_prompt.format(goals=checker.persona,context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
    state["messages"] = [answer.content]
    return state

def save_search(state: CheckingState):
    """ Save interviews """
    messages = state["messages"]
    results = get_buffer_string(messages)
    state["results"] = results
    return state

def route_messages(state: CheckingState):
    """ Route between question and answer """
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage)]
    )
    if num_responses >= max_num_turns:
        return 'save_search'
    
    if len(messages) < 2:  # Check if the messages list has at least two elements
      return "search_query"
    
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'save_search'
    return "search_query"

reason_prompt = """Given the data in the document as following\n\n{fact} \n\nfind the reason for its correctness using\n\n {context}"""
def state_reason(state: CheckingState):
    facts = state["fact"]
    context = state["context"]
    checker = state["fchecker"]
    system_prompt = reason_prompt.format(fact=facts,topic=context)
    correction = llm.invoke([SystemMessage(content=system_prompt)]+[HumanMessage(content=f"Give the reason why this fact is correct or incorrect.")])
    state["reason"] = correction.content
    return state


web_search_builder = StateGraph(CheckingState)
web_search_builder.add_node("search_query", generate_question)
web_search_builder.add_node("web_search",search_web)


web_search_builder.add_node("fact_correctness",generate_answer)
web_search_builder.add_node("save_search",save_search)
web_search_builder.add_node("reason_writer",state_reason)
#Logic of Web Search
web_search_builder.add_edge(START,"search_query")
web_search_builder.add_edge("search_query","web_search")

web_search_builder.add_edge("web_search","fact_correctness")
web_search_builder.add_conditional_edges("fact_correctness",route_messages,["search_query","save_search"])
web_search_builder.add_edge("save_search","reason_writer")
web_search_builder.add_edge("reason_writer",END)


# memory = MemorySaver()
web_search_graph = web_search_builder.compile(checkpointer=memory).with_config(run_name="Web search")
# image_bytes = web_search_graph.get_graph(xray=1).draw_mermaid_png()
# image = PILImage.open(io.BytesIO(image_bytes))
# image.save('web_search.png')

topic = """The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data."""

# for event in graph.stream({"topic":topic, "max_checker":5}, thread, stream_mode="values"):    # Review
#     checkers = event.get('checker', '')
#     if checkers:
#         for analyst in checkers:
#             print(f"conference: {analyst.conference}")
#             print(f"Websites: {analyst.Websites}")
#             print(f"Description: {analyst.description}")
#             print("-" * 50) 


messages = [HumanMessage(f"So you said you were checking an article on {topic}?")]
thread = {"configurable": {"thread_id": "1"}}
# interview = web_search_graph.invoke({"fchecker": checkers[0],"fact":topic ,"messages": messages, "max_num_turns": 2}, thread)
# print(interview['reason'][0])


def initiate_all_checker(state: FactCheckerState):
    topic = state["topic"]
    return [Send("Check_facts", {"checker": checker, "messages": [HumanMessage(content=f"so you said that will fact check {topic}?")]}) for checker in state["fchecker"]]

reason_prompt = """given the data in the document as following\n\n{fact} \n\nfind the reason for its correctness using\n\n {context}
                """

def state_reason(state: FactCheckerState):
    facts = state["Facts"]
    context = state["context"]
    system_prompt = reason_prompt.format(topic=context,fact=facts)
    correction = llm.invoke([SystemMessage(content=system_prompt)]+[HumanMessage(content=f"Give the reason why this fact is correct or incorrect.")])
    state["reason"] = correction.content
    return state


sanity_prompt = """given the context in the document as following:\n\n{facts}
                \n\n and now verify this based on {reason} 
                """

def Decide_sanity(state: FactCheckerState):
    facts = state["Facts"]
    reason = state["reason"]
    system_prompt = sanity_prompt.format(topic=facts,fact=reason)
    decision = llm.invoke([SystemMessage(content=system_prompt)]+[HumanMessage(content=f"give boolean decision")])
    state["Decision"] = decision.content
    return state

def Score_document(state: FactCheckerState):
    pass

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
image_bytes = Checker_graph.get_graph(xray=1).draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_bytes))
image.save('fact_checker.png')

max_checker = 3 
topic = """The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data."""

thread = {"configurable": {"thread_id": "1"}}

# for event in Checker_graph.stream({"topic":topic,"max_checker":max_checker,}, thread, stream_mode="values"):
#     checker = event.get('checker', '')
#     if checker:
#         for ch in checker:
#             print(f"Conference: {ch.conference}")
#             print(f"Websites: {ch.Websites}")
#             print(f"Description: {ch.description}")
#             print("-" * 50) 

# for event in Checker_graph.stream(None, thread, stream_mode="updates"):
#     print("--Node--")
#     node_name = next(iter(event.keys()))
#     print(node_name)