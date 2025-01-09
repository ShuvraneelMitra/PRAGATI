from create_questions.schemas import Reviewer, QuestionState
from schemas import AnswerState, IntermediateAnswerState, Queries
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import copy

from create_questions.helper import invoke_llm

## Idea:
### 1. We assume that the LLM would not be able to directly answer a reasoning based Yes/No question. Hence the question would be broken into sub-queries.
### 2. The sub-queries would be generated by the LLM. These sub-queries would be used as search queries for retrieval (make use of LLM to form 2 versions of each sub-query).
### 3. Form an intermediate answer for each of the sub-queries. 
### 4. Provide these Q&A pairs to the LLM as reference to form the final answer.

## The transition from QuestionState to AnswerState would be done by an intermediate `compile_questions` node.
## This is because we would not require information about individual reviewers after generation of questions.

def generate_sub_queries(question:str) -> list:
    """
    Generate sub-queries for the given question
    """
    return []

def retrieve_references(state:AnswerState) -> IntermediateAnswerState:
    messages = state['messages']
    paper = state['paper']
    topic = state['topic']
    questions = state['questions']
    conference = state['conference']

    updated_state = copy.deeptcopy(state)

    questions_updated = []
    for question in questions:
        sub_queries = generate_sub_queries(question)
        question_updated = Queries(original_query=question, sub_queries=sub_queries)
        questions_updated.append(question_updated)

    



