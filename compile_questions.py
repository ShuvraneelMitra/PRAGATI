import copy
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils.schemas import Reviewer, Queries, QAPair
from utils.chat import invoke_llm_langchain
from agentstates import QuestionState, AnswerState

def compile_questions(state:QuestionState) -> AnswerState:
    """
    Compile all questions made by all the reviewers into a single list
    """
    updated_state = AnswerState(messages=state['messages'],
                                paper=state['paper'],
                                topic=state['topic'],
                                questions=[],
                                conference=state['conference'],
                                conference_description=state['conference_description'],
                                answer=None,
                                token_usage=state['token_usage'])
    for reviewer in state['reviewers']:
        updated_state['questions'].extend(reviewer['questions'])

    return updated_state