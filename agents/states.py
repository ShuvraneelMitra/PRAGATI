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
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(None,
                                                                     description="List of messages related to the question state")
    paper: Paper = Field(None, description="Paper related to the question state")
    topic: str = Field(None, description="Topic related to the question state")
    num_reviewers: int = Field(None, description="Number of reviewers to be selected")
    reviewers: List[Reviewer] = Field(None, description="List of reviewers selected")
    token_usage: TokenTracker = Field(None, description="Token usage for the question state")
    questions: Annotated[List[str], add] = Field(None, description="List of queries related to the question state")
    model_config = {
        "validate_assignment": True
    }


class AnswerState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(None,
                                                                     description="List of messages related to the "
                                                                                 "answer state")
    paper: Paper = Field(None, description="Paper related to the answer state")
    topic: str = Field(None, description="Topic related to the answer state")
    questions: Annotated[List[str], add] = Field(None, description="List of questions related to the answer state")
    answers: Annotated[List[str], add] = Field(None, description="Final 'YES' or 'NO' answer to each question")
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
    token_usage: TokenTracker = Field(None, description="Token usage for the intermediate answer state")
    answers: Annotated[List[str], add] = Field(None, description="Final 'YES' or 'NO' answer to each question")
    model_config = {
        "validate_assignment": True  # Allows field updates with validation
    }


class SingleQuery(BaseModel):
    question: str = Field(None, description="Single query to be asked")
    sub_queries: Annotated[List[QAPair], add] = Field(None,
                                                      description="List of sub queries related to the single query")
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
