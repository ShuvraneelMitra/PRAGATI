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
    questions = []
    for reviewer in state.reviewers:
        questions.extend(reviewer.questions)

    updated_state = AnswerState(messages=state.messages,
                                paper=state.paper,
                                questions=questions,
                                conference=state.conference,
                                conference_description=state.conference_description,
                                )

    return updated_state

def compile2(state:AnswerState) -> AnswerState:
    """
    Compile all questions made by all the reviewers into a single list
    """

    updated_state = AnswerState(paper=state.paper,
                                questions=state.questions,
                                conference=state.conference,
                                conference_description=state.conference_description,
                                )

    return updated_state