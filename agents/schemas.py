from pydantic import BaseModel, Field
from typing import List, Any, Dict, Sequence

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class Reviewer(BaseModel):
    id: int = Field(None, description="ID of the reviewer")
    specialisation: str = Field(None, description="Specialisation of the reviewer")
    questions: List[str] = Field(None, description="Questions asked by the reviewer")


class QAPair(BaseModel):
    query: str = Field(None, description="Query string")
    answer: str = Field(None, description="Answer string")
    references: List[str] = Field(
        None, description="List of references related to the query and answer"
    )


class Paper(BaseModel):
    title: str = Field(None, description="Title of the paper")
    topic: str = Field(None, description="Topic of the paper")
    filename: str = Field(None, description="Filename of the paper")
    sections: Sequence[str] = Field(
        None,
        description="A list of the sections present in the paper such as Abstract, "
        "Introduction etc.",
    )

class FRPair(BaseModel):
    """Fact-Reference Pair representing a claim and its verification"""
    claim: str = Field(description="The original claim to be fact-checked")
    search_query: str = Field(default="", description="Query used for searching")
    k_search_query: str = Field(default="", description="Query used for knowledge search")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from web search")
    k_search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from knowledge search")
    verification: Dict[str, Any] = Field(default_factory=dict, description="Verification results with scores")
    k_verification: Dict[str, Any] = Field(default_factory=dict, description="Verification results with scores from knowledge search")