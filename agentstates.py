from pydantic import BaseModel, Field
from typing import List, Optional

from utils.schemas import TokenTracker, Reviewer, Queries, QAPair

class QuestionState(BaseModel):
    messages: List[str]
    paper: str ## up for decision
    topic: str
    num_reviewers: int
    reviewers: List[Reviewer]
    conference: str
    conference_description: str
    token_usage: TokenTracker

class AnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the answer state")
    paper: str = Field(None, description="Paper related to the answer state")
    topic: str = Field(None, description="Topic related to the answer state")
    questions: List[str] = Field(None, description="List of questions related to the answer state")
    conference: str = Field(None, description="Conference related to the answer state")
    conference_description: str = Field(None, description="Description of the conference")
    answer: str = Field(None, description="Final 'YES' or 'NO' answer to the question")
    token_usage: TokenTracker = Field(None, description="Token usage for the answer state")

class IntermediateAnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the intermediate answer state")
    paper: str = Field(None, description="Paper related to the intermediate answer state")
    topic: str = Field(None, description="Topic related to the intermediate answer state")
    questions: List[Queries] = Field(None, description="List of queries related to the intermediate answer state")
    conference: str = Field(None, description="Conference related to the intermediate answer state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the intermediate answer state")

