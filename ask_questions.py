import copy
from typing import List
import os
import pprint
import logging 

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utils.schemas import Reviewer, Queries, QAPair, Paper
from utils.chat import invoke_llm_langchain
from utils.helper import print_reviewers
from agentstates import QuestionState

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def create_researchers(state:QuestionState) -> QuestionState:
    """
    Create a list of reviewers for the given conference, each with a specialisation
    """
    messages = state.messages
    num_reviewers = state.num_reviewers
    conference = state.conference
    conference_description = state.conference_description

    researcher_creation_messages = []
    load_dotenv()   
    def search_conference(conference: str) -> str:
        """
        Search for the description of the conference. Uses Web Search for the same
        """
        return "Conference Description"
    
    if conference_description is None:
        conference_description = search_conference(conference)

    researcher_creation_prompt = f"""
    You are a helpful assistant. You need to give a list of {num_reviewers} reviewers, with proper characters as will be provided for the popular scientific conference `{conference}`.

    Following is a brief description of the conference:
    {conference_description}

    Follow the given instructions carefully:
    1. Each reviewer must be a distinguished researcher in the topic of the given conference. 

    2. Each reviewer should have an area of specialisation, which must be a sub-field of the topic which is discussed in the conference. Carefully go through the conference description provided to decide the areas of specialisation of the reviewers.

    3. Return your answer ONLY as a list of JSONs with length {num_reviewers}. Each JSON in the list should have the following keys:
    -- 'id': a unique number between 1 and {num_reviewers}
    -- 'specialisation': Area of specialisation of the reviewer. This area of specialisation should be coherent with the topics of the given conference.

    There should be no extra verbiage in your answer. Only the list of JSONs should be returned. 

    DO NOT return any text other than the list of JSONs.
    """

    researcher_creation_messages.append(HumanMessage(content=researcher_creation_prompt))

    messages, input_tokens, output_tokens = invoke_llm_langchain(researcher_creation_messages, api_key=api_key)
    response = messages[-1].content

    reviewers = eval(response.replace("\n", "").replace("```json", "").replace("```", ""))
    reviewers = [Reviewer(**reviewer) for reviewer in reviewers]

    # pprint.pprint(reviewers)

    for i in range(num_reviewers):
        reviewers[i].id = i+1

    state.reviewers = reviewers

    return state

def get_questionnaire(state:QuestionState) -> QuestionState:
    """
    Generate a list of questions for each reviewer
    """
    messages = state.messages
    num_reviewers = state.num_reviewers
    conference = state.conference
    conference_description = state.conference_description
    reviewers = state.reviewers
    topic = state.paper.title

    questions_system_prompt = f"""
    There is a research paper being reviewed for publication in a conference. You are a helpful assistant tasked with creating a list of questions for the reviewers to ask.

    You will be provided the specialisation of the reviewers one by one. You need to generate a list of quesions to be asked by the reviewers. 

    The conference is `{conference}`. The topic of the paper is `{topic}`. The conference description is as follows:
    ```
    {conference_description}
    ```
    You shall be provided the Abstract and the Conclusion of the paper. Refer to these and ensure that the questionnaire exhaustively covers all topics of the paper. 

    Follow the given instructions carefully:
    -- The questions should be coherent with the topic of the conference and the specialisation of the reviewers, which shall be provided.
    -- The questions should have a binary answer (Yes/No) and should be relevant to the topic of the paper.
    -- The questions should be formed in such a way that a 'Yes' answer would indicate a positive review and a 'No' answer would indicate a negative review. That is, 'Yes' answers should indicate that the paper is good and 'No' answers should indicate that the paper is not good.
    -- The entire list if questions should effectively help the reviewers in evaluating the paper.
    -- Keep the number of questions less than 3. Do not exceed this limit.
    -- DO NOT repeat the questions. Each question should be unique and should cover a different aspect of the paper.

    """
    questions_messages = [SystemMessage(content=questions_system_prompt)]

    ## add abstract and conclusion as messages (HumanMessage)

    ## replace this part with parallel processing
    for reviewer in reviewers:
        questions = get_questions_for_reviewer(questions_messages,reviewer,conference,topic)
        reviewer.questions = questions
    
    
    return state

def get_questions_for_reviewer(messages,reviewer,conference,topic):
    """
    Generate a list of questions for a given reviewer
    """
    reviewer_specialisation = reviewer.specialisation

    questions_prompt = f"""
    Now you need to generate a list of questions for the reviewer with specialisation `{reviewer_specialisation}`. The conference is `{conference}` and the topic of the paper is `{topic}`.

    Keep in mind the previously mentioned instructions while generating the questions. The questions should be coherent with the topic of the conference and the specialisation of the reviewer.

    Cover all aspects of the paper in the questions. The questions should be Yes/No questions, with 'Yes' denoting positive output.

    Return your reponse as a Python list of strings, each string being a question. Keep proper syntax, like start with `[` and end with `]` and ensure each string in the list is in-between quotes.
    Return only the list of questions. Do not include any extra verbiage in your response.
    """
    messages.append(HumanMessage(content=questions_prompt))

    messages, input_tokens, output_tokens = invoke_llm_langchain(messages, api_key=api_key)
    response = messages[-1].content

    # print(response)
    # print()

    questions = eval(response.replace("\n", "").replace("```python", "").replace("```", ""))

    return questions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    filename='kdsh-2025.log')
    logging.captureWarnings(True)

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

    compiler = StateGraph(QuestionState)
    memory = MemorySaver()

    compiler.add_node("create_researchers", create_researchers)
    compiler.add_node("get_questionnaire", get_questionnaire)

    compiler.add_edge(START, "create_researchers")
    compiler.add_edge("create_researchers", "get_questionnaire")
    compiler.add_edge("get_questionnaire", END)

    graph = compiler.compile()

    thread = {
                "configurable": 
                    {
                        "thread_id": os.getenv("TEMP_PAPER_OBJ_ID")
                    },
             }
    final_state = {}

    for event in graph.stream(initial_state, thread):
        for key, value in event.items():
            print("Output from node: ", end=' ')
            pprint.pprint(key)
            pprint.pprint("-------------------------")
            # pprint.pprint(value['reviewers'])
            print_reviewers(value['reviewers'])
            final_state = value

    logging.info("Final State:")
    logging.info(final_state)

    pprint.pprint("Final State:")
    pprint.pprint(final_state)
