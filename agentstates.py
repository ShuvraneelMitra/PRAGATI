from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from langgraph.graph import MessagesState
from typing_extensions import TypedDict
import operator
from decimal import Decimal

from utils.schemas import TokenTracker, Reviewer, Queries, QAPair, Paper

## QuestionState, AnswerState and IntermediateAnswerState are defined on paper-level basis

class QuestionState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the question state")
    paper: Paper = Field(None, description="Paper related to the question state")
    num_reviewers: int = Field(None, description="Number of reviewers to be created")
    reviewers: List[Reviewer] = Field(None, description="List of reviewers")
    conference: str = Field(None, description="Conference related to the question state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the question state")

class AnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the answer state")
    paper: Paper = Field(None, description="Paper related to the answer state")
    questions: List[str] = Field(None, description="List of questions related to the answer state")
    conference: str = Field(None, description="Conference related to the answer state")
    conference_description: str = Field(None, description="Description of the conference")
    answers: str = Field(None, description="Final 'YES' or 'NO' answer to the question")
    token_usage: TokenTracker = Field(None, description="Token usage for the answer state")
    publishable: bool = Field(None, description="Whether the paper is publishable")

class IntermediateAnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the intermediate answer state")
    paper: Paper = Field(None, description="Paper related to the intermediate answer state")
    questions: List[Queries] = Field(None, description="List of queries related to the intermediate answer state")
    conference: str = Field(None, description="Conference related to the intermediate answer state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the intermediate answer state")

class Factchecker(BaseModel):
    conference: str = Field(
        description="Fact checker conference affiliation which will be any out of CVPR, EMNLP, KDD, NeurIPS, TMLR",
    )
    Websites: str = Field(
        description="Websites he will refer to for facts refer any of or multiple among Arxiv, Wikipedia, Google Scholar (use url) or anything similar ",
    )
    description: str = Field(
        description="Description of the fact checker focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Conference: {self.conference}\nWebsites: {self.Websites}\nDescription: {self.description}\n"
    def web(self) -> str:
        return self.Websites
    class Config:
        arbitrary_types_allowed = True

class Perspectives(BaseModel):

    checker: List[Factchecker] = Field(
        description="Comprehensive list of checker with their affiliations and methods.",
    )
    class Config:
        arbitrary_types_allowed = True

class Generatecheckertate(TypedDict):
    topic: str # Research topic
    max_checker: int # Number of Factchecker
    checker: List[Factchecker] # checker asking questions

class CheckingState(MessagesState):
    max_num_turns: int
    fchecker: Factchecker
    fact: str
    messages: list
    context: Annotated[list, operator.add]
    results: str
    class Config:
        arbitrary_types_allowed = True

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search Query for retrieval")
    class Config:
      arbitrary_types_allowed = True

class FactCheckerState(TypedDict):
    topic: str #whether the fact stated is correct or not 
    max_checker: int #number of parallel checkers working on that part
    fchecker: List[Factchecker]
    Facts: list
    context: str
    Decision: bool
    reason: str
    sanity_score: Decimal 

