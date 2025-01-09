from pydantic import BaseModel, Field
from typing import List, Optional

class AnswerState(BaseModel):
    messages: List[str]
    paper: str
    topic: str
    questions: List[str]
    conference: str
    conference_description: str
    answers: List[str]

class Queries(BaseModel):
    original_query: str
    sub_queries: List[str]

class IntermediateAnswerState(BaseModel):
    messages: List[str]
    paper: str
    topic: str
    questions: List[Queries]
    conference: str
    conference_description: str
    answers: List[str]

