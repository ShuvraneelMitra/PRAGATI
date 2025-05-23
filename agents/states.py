from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add
from typing import Dict, Any
from agents.schemas import TokenTracker, QAPair, Reviewer, Paper, FRPair


class SingleQuery(BaseModel):
    question: str = Field(None, description="Single query to be asked")
    sub_queries: Sequence[QAPair] = Field(
        None, description="Sequence of sub queries related to the single query"
    )
    answer: str = Field(None, description="Yes/No Answer to the single query")
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class QuestionState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        None, description="Sequence of messages related to the question state"
    )
    paper: Paper = Field(None, description="Paper related to the question state")
    num_reviewers: int = Field(4, description="Number of reviewers to be selected")
    num_subqueries: int = Field(
        3, description="Number of subqueries per broad question"
    )
    reviewers: Sequence[Reviewer] = Field(
        default_factory=list, description="Sequence of reviewers selected"
    )
    token_usage: TokenTracker = Field(
        None, description="Token usage for the question state"
    )
    queries: Sequence[Sequence[SingleQuery]] = Field(
        None, description="Sequence of compiled questions with answers"
    )
    publishability: str = Field(None, description="The final assessment")
    suggestions: str = Field(None, description="The final suggestions")
    model_config = ConfigDict(validate_assignment=True)

    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return self is other

class FactCheckerState(BaseModel):
    paper: Paper = Field(None, description="Paper that should be fact-checked")
    inputs: str = Field(None, description="Original text chunk to be fact-checked")
    search_query: str = Field(
        None, description="Generated search query for fact-checking"
    )
    web_search_results: Sequence[str] = Field(
        None, description="Results retrieved from web search"
    )
    pairs: Annotated[Sequence[FRPair], add] = Field(
        None, description="Pairs of text and facts from web search"
    )
    kpairs: Annotated[Sequence[FRPair], add] = Field(
        None, description="Pairs of text and facts from Knowledge search"
    )
    token_usage: TokenTracker = Field(
        int, description="Token usage for fact-checking process"
    )
    is_factual: bool = Field(
        None, description="Whether the input text is factual or not"
    )
    errors: str = Field(None, description="Error message if any")
    current_index: int = Field(0, description="Index of the current fact pair")
    no_claims: int = Field(0, description="Number of claims made")
    claim_scores: Sequence[int] = Field(
        default_factory=list, description="Scores for each claim made"
    )
    total_score: int = Field(0, description="Total score of the fact-checking process")
    average_score: float = Field(
        0.0, description="Average score of the fact-checking process"
    )
    model_config = {"validate_assignment": True}
    # class Config:
    #     frozen = True


class CombinedPaperState(BaseModel):
    fact_checker_results: FactCheckerState = Field(
        None, description="Results from fact checker workflow"
    )
    qa_results: QuestionState = Field(
        None, description="Results from question-answering workflow"
    )
    overall_assessment: Optional[str] = Field(
        None, description="Combined assessment based on both workflows"
    )
    is_reliable: Optional[bool] = Field(
        None, description="Whether the paper is considered reliable"
    )
    model_config = ConfigDict(validate_assignment=True)
    # class Config:
    #     frozen = True

class PaperReview(BaseModel):
    factual: bool = Field(
        None, description="Whether the paper is factual or not"
    )
    fact_checker_score: float = Field(
        None, description="Score from the fact-checker"
    )
    Publishability: str = Field(
        None, description="Publishability score from the QA"
    )
    Suggestions: str = Field(
        None, description="Suggestions from the QA"
    )
    
class PaperState(BaseModel):
    fact_checker_results: FactCheckerState = Field(
        None, description="Results from fact checker workflow"
    )
    qa_results: QuestionState = Field(
        None, description="Results from question-answering workflow"
    )
    overall_assesment: PaperReview = Field(
        default_factory=PaperReview, description="Combined assessment based on both workflows"
    )