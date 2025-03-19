from pydantic import BaseModel, Field
from typing import Annotated, Sequence
from operator import add
from agents.schemas import TokenTracker, FRPair

class FactCheckerState(BaseModel):
    inputs: str = Field(None, description="Original text chunk to be fact-checked")
    search_query: str = Field(None, description="Generated search query for fact-checking")
    web_search_results: Sequence[str] = Field(None, description="Results retrieved from web search")
    pairs: Annotated[Sequence[FRPair], add] = Field(None, description="Pairs of text and facts from web search")
    kpairs: Annotated[Sequence[FRPair], add] = Field(None, description="Pairs of text and facts from Knowledge search")
    token_usage: TokenTracker = Field(int, description="Token usage for fact-checking process")
    is_factual: bool = Field(None, description="Whether the input text is factual or not")
    errors: str = Field(None, description="Error message if any")
    current_index: int = Field(0, description="Index of the current fact pair")
    no_claims: int = Field(0, description="Number of claims made")
    total_score: int = Field(0, description="Total score of the fact-checking process")
    average_score: float = Field(0.0, description="Average score of the fact-checking process")
    model_config = {
        "validate_assignment": True
    }