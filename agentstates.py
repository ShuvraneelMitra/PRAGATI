from pydantic import BaseModel, Field
from typing import List, Optional

from utils.schemas import TokenTracker, Reviewer, Queries, QAPair, Paper

## QuestionState, AnswerState and IntermediateAnswerState are defined on paper basis

class QuestionState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the question state")
    paper: Paper ##object id of the paper
    num_reviewers: int
    reviewers: List[Reviewer]
    conference: str
    conference_description: str
    token_usage: TokenTracker

class AnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the answer state")
    paper: Paper = Field(None, description="Paper related to the answer state")
    questions: List[str] = Field(None, description="List of questions related to the answer state")
    conference: str = Field(None, description="Conference related to the answer state")
    conference_description: str = Field(None, description="Description of the conference")
    answer: List[str] = Field(None, description="Final 'YES' or 'NO' answer to the question")
    token_usage: TokenTracker = Field(None, description="Token usage for the answer state")
    publishable: bool = Field(None, description="Whether the paper is publishable")

class IntermediateAnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the intermediate answer state")
    paper: Paper = Field(None, description="Paper related to the intermediate answer state")
    questions: List[Queries] = Field(None, description="List of queries related to the intermediate answer state")
    conference: str = Field(None, description="Conference related to the intermediate answer state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the intermediate answer state")

