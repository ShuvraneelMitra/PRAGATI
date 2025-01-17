from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Sequence
from langgraph.graph import MessagesState
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator
from operator import add
from decimal import Decimal

from utils.schemas import TokenTracker, Reviewer, Queries, QAPair, Paper

class QuestionState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(None, description="List of messages related to the question state")  
    paper: Paper = Field(None, description="Paper related to the question state") 
    topic: str = Field(None, description="Topic related to the question state")
    num_reviewers: int = Field(None, description="Number of reviewers to be selected")
    reviewers: List[Reviewer] = Field(None, description="List of reviewers selected")
    conference: str = Field(None, description="Conference related to the question state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the question state")
    questions: Annotated[List[str], add] = Field(None, description="List of queries related to the question state")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }

class AnswerState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(None, description="List of messages related to the answer state")
    paper: Paper = Field(None, description="Paper related to the answer state")
    topic: str = Field(None, description="Topic related to the answer state")
    questions: Annotated[List[str], add] = Field(None, description="List of questions related to the answer state")
    conference: str = Field(None, description="Conference related to the answer state")
    conference_description: str = Field(None, description="Description of the confere    reviewers = state.reviewersnce")
    answers: Annotated[List[str],add] = Field(None, description="Final 'YES' or 'NO' answer to each question")
    token_usage: TokenTracker = Field(None, description="Token usage for the answer state")
    publishable: bool = Field(None, description="Whether the paper is publishable or not")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }
 
class IntermediateAnswerState(BaseModel):
    messages: List[str] = Field(None, description="List of messages related to the intermediate answer state")
    paper: Paper = Field(None, description="Paper related to the intermediate answer state")
    topic: str = Field(None, description="Topic related to the intermediate answer state")
    queries: List[Queries] = Field(None, description="List of queries related to the intermediate answer state")
    conference: str = Field(None, description="Conference related to the intermediate answer state")
    conference_description: str = Field(None, description="Description of the conference")
    token_usage: TokenTracker = Field(None, description="Token usage for the intermediate answer state")
    answers: Annotated[List[str], add] = Field(None, description="Final 'YES' or 'NO' answer to each question")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }

class SingleQuery(BaseModel):
    question: str = Field(None, description="Single query to be asked")
    sub_queries: Annotated[List[QAPair], add] = Field(None, description="List of sub queries related to the single query")
    answer: str = Field(None, description="Yes/No Answer to the single query")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }

class QueryRoutingState(BaseModel):
    query: Queries = Field(None, description="Query to be routed")
    conference: str = Field(None, description="Conference related to the query")
    conference_description: str = Field(None, description="Description of the conference")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }

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
    
    @property
    def web(self) -> str:
        return self.Websites
    
    class Config:
        arbitrary_types_allowed = True

class Perspectives(BaseModel):
    checker: List[Factchecker] = Field(description="Comprehensive list of checker with their affiliations and methods.")
    
    class Config:
        arbitrary_types_allowed = True

class GenerateCheckerState(TypedDict):
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

