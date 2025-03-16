from pydantic import BaseModel, Field
from typing import Annotated, Sequence
from operator import add
from schemas import TokenTracker, FRPair

class FactCheckerState(BaseModel):
    input_text: str = Field(None, description="Original text chunk to be fact-checked")
    search_query: str = Field(None, description="Generated search query for fact-checking")
    web_search_results: Sequence[str] = Field(None, description="Results retrieved from web search")
    fact_pairs: Annotated[Sequence[FRPair], add] = Field(None, description="Pairs of text and facts from web search")
    token_usage: TokenTracker = Field(None, description="Token usage for fact-checking process")
    is_factual: bool = Field(None, description="Whether the input text is factual or not")
    model_config = {
        "validate_assignment": True
    }