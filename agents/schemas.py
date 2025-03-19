from pydantic import BaseModel, Field
from typing import List, Any, Dict

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class FRPair(BaseModel):
    """Fact-Reference Pair representing a claim and its verification"""
    claim: str = Field(description="The original claim to be fact-checked")
    search_query: str = Field(default="", description="Query used for searching")
    k_search_query: str = Field(default="", description="Query used for knowledge search")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from web search")
    k_search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from knowledge search")
    verification: Dict[str, Any] = Field(default_factory=dict, description="Verification results with scores")
    k_verification: Dict[str, Any] = Field(default_factory=dict, description="Verification results with scores from knowledge search")
    