import copy
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils.schemas import Reviewer, Queries, QAPair
from utils.chat import invoke_llm_langchain
from agentstates import QuestionState, AnswerState, IntermediateAnswerState

def create_researchers(state:QuestionState) -> QuestionState:
    """
    Create a list of reviewers for the given conference, each with a specialisation
    """
    messages = state['messages']
    num_reviewers = state['num_reviewers']
    conference = state['conference']
    conference_description = state['conference_description']
    topic = state['topic']

    researcher_creation_messages = []

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
    """

    researcher_creation_messages.append(HumanMessage(content=researcher_creation_prompt))

    response, input_tokens, output_tokens = invoke_llm_langchain(researcher_creation_messages)

    reviewers = eval(response.replace("\n", "").replace("```json", "").replace("```", ""))

    messages.append(HumanMessage(content=researcher_creation_prompt))
    messages.append(AIMessage(content=response))

    for i in range(num_reviewers):
        reviewers['id'] = i+1

    state['reviewers'] = reviewers

    return state

def get_questionnaire(state:QuestionState) -> QuestionState:
    """
    Generate a list of questions for each reviewer
    """
    messages = state['messages']
    num_reviewers = state['num_reviewers']
    conference = state['conference']
    conference_description = state['conference_description']
    topic = state['topic']
    reviewers = state['reviewers']

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
    -- Keep the number of questions between 3 and 5. Do not exceed this limit.
    -- DO NOT repeat the questions. Each question should be unique and should cover a different aspect of the paper.

    """
    questions_messages = [SystemMessage(content=questions_system_prompt)]

    ## add abstract and conclusion as messages (HumanMessage)

    ## replace this part with parallel processing
    for reviewer in reviewers:
        questions = get_questions_for_reviewer(questions_messages,reviewer,conference,topic)
        reviewer['questions'] = questions
    
    
    return state

def get_questions_for_reviewer(messages,reviewer,conference,topic):
    """
    Generate a list of questions for a given reviewer
    """
    reviewer_specialisation = reviewer['specialisation']

    questions_prompt = f"""
    Now you need to generate a list of questions for the reviewer with specialisation `{reviewer_specialisation}`. The conference is `{conference}` and the topic of the paper is `{topic}`.

    Keep in mind the previously mentioned instructions while generating the questions. The questions should be coherent with the topic of the conference and the specialisation of the reviewer.

    Cover all aspects of the paper in the questions. The questions should be Yes/No questions, with 'Yes' denoting positive output.

    Return your reponse as a Python list of strings, each string being a question. Keep proper syntax, like start with `[` and end with `]` and ensure each string in the list is in-between quotes.
    Return only the list of questions. Do not include any extra verbiage in your response.
    """
    messages.append(HumanMessage(content=questions_prompt))

    response, input_tokens, output_tokens = invoke_llm_langchain(messages)

    questions = eval(response.replace("\n", "").replace("```python", "").replace("```", ""))

    return questions

## Idea:
### 1. We assume that the LLM would not be able to directly answer a reasoning based Yes/No question. Hence the question would be broken into sub-queries.
### 2. The sub-queries would be generated by the LLM. These sub-queries would be used as search queries for retrieval (make use of LLM to form 2 versions of each sub-query).
### 3. Form an intermediate answer for each of the sub-queries. 
### 4. Provide these Q&A pairs to the LLM as reference to form the final answer.

## The transition from QuestionState to AnswerState would be done by an intermediate `compile_questions` node.
## This is because we would not require information about individual reviewers after generation of questions.

def generate_sub_queries(state:AnswerState) -> IntermediateAnswerState:
    """
    Generate sub-queries for each question in the state
    """
    messages = state['messages']
    paper = state['paper']
    topic = state['topic']
    questions = state['questions']
    conference = state['conference']

    updated_state = copy.deepcopy(state)

    def generate_sub_queries_by_question(question:str) -> List[QAPair]:
        """
        Generate sub-queries for a given question
        """
        sub_queries = []
        return sub_queries

    questions_updated = []
    for question in questions:
        sub_queries = generate_sub_queries_by_question(question)
        question_updated = Queries(original_query=question, sub_queries=sub_queries)
        questions_updated.append(question_updated)

    updated_state['questions'] = questions_updated
    return updated_state

def retrieve_references(state:IntermediateAnswerState) -> IntermediateAnswerState:
    """
    Retrieve references for each sub-query in the state
    """
    messages = state['messages']
    paper = state['paper']
    topic = state['topic']
    questions = state['questions']
    conference = state['conference']

    updated_state = copy.deepcopy(state)

    questions_updated = []
    for question in questions:
        pass

    return updated_state

def answer_sub_queries(state:IntermediateAnswerState) -> IntermediateAnswerState:
    """
    Answer sub-queries for each question in the state
    """
    messages = state['messages']
    paper = state['paper']
    topic = state['topic']
    questions = state['questions']
    conference = state['conference']

    updated_state = copy.deepcopy(state)

    questions_updated = []
    for question in questions:
        pass

    return updated_state

def summarise_results(state:IntermediateAnswerState) -> AnswerState:
    """
    Summarise the results of the sub-queries into a final Yes/No answer
    """
    messages = state['messages']
    paper = state['paper']
    topic = state['topic']
    questions = state['questions']
    conference = state['conference']

    updated_state = copy.deepcopy(state)

    questions_updated = []
    for question in questions:
        pass

    return updated_state

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



    

    



