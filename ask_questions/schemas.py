from pydantic import BaseModel, Field
from typing import List, Optional

class AnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the answer state")
    paper: str = Field(None, description="Paper related to the answer state")
    topic: str = Field(None, description="Topic related to the answer state")
    questions: List[str] = Field(None, description="List of questions related to the answer state")
    conference: str = Field(None, description="Conference related to the answer state")
    conference_description: str = Field(None, description="Description of the conference")
    answer: str = Field(None, description="Final 'YES' or 'NO' answer to the question")

class QAPair(BaseModel):
    query: str = Field(None, description="Query string")
    answer: str = Field(None, description="Answer string")
    references: List[str] = Field(None, description="List of references related to the query and answer")

class Queries(BaseModel):
    original_query: str = Field(None, description="Original query string")
    sub_qas: List[QAPair] = Field(None, description="List of question-answer pairs related to the original query")

class IntermediateAnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the intermediate answer state")
    paper: str = Field(None, description="Paper related to the intermediate answer state")
    topic: str = Field(None, description="Topic related to the intermediate answer state")
    questions: List[Queries] = Field(None, description="List of queries related to the intermediate answer state")
    conference: str = Field(None, description="Conference related to the intermediate answer state")
    conference_description: str = Field(None, description="Description of the conference")

