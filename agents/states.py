from pydantic import BaseModel, Field
from typing import Optional, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add

from agents.schemas import TokenTracker, Reviewer, QAPair, Paper


class QuestionState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        None, description="Sequence of messages related to the " "question state"
    )
    paper: Paper = Field(None, description="Paper related to the question state")
    topic: str = Field(None, description="Topic related to the question state")
    num_reviewers: int = Field(4, description="Number of reviewers to be selected")
    reviewers: Sequence[Reviewer] = Field(
        None, description="Sequence of reviewers selected"
    )
    token_usage: TokenTracker = Field(
        None, description="Token usage for the question state"
    )
    questions: Annotated[Sequence[str], add] = Field(
        None, description="Sequence of queries related to the question " "state"
    )
    model_config = {"validate_assignment": True}


class AnswerState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        None, description="Sequence of messages related to the " "answer state"
    )
    paper: Paper = Field(None, description="Paper related to the answer state")
    topic: str = Field(None, description="Topic related to the answer state")
    questions: Annotated[Sequence[str], add] = Field(
        None, description="Sequence of questions related to the answer " "state"
    )
    answers: Annotated[Sequence[str], add] = Field(
        None, description="Final 'YES' or 'NO' answer to each question"
    )
    token_usage: TokenTracker = Field(
        None, description="Token usage for the answer state"
    )
    publishable: bool = Field(
        None, description="Whether the paper is publishable or not"
    )
    model_config = {"validate_assignment": True}


class SingleQuery(BaseModel):
    question: str = Field(None, description="Single query to be asked")
    sub_queries: Annotated[Sequence[QAPair], add] = Field(
        None, description="Sequence of sub queries related to the single " "query"
    )
    answer: str = Field(None, description="Yes/No Answer to the single query")
    model_config = {"validate_assignment": True}
