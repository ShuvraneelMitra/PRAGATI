from pydantic import BaseModel, Field
from typing import List, Optional

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class Reviewer(BaseModel):
    id: int = Field(description="ID of the reviewer")
    specialisation: str = Field(description="Specialisation of the reviewer")
    questions: List[str] = Field(description="Questions asked by the reviewer")


class QAPair(BaseModel):
    query: str = Field(None, description="Query string")
    answer: str = Field(None, description="Answer string")
    references: List[str] = Field(None, description="List of references related to the query and answer")

class Queries(BaseModel):
    original_query: str = Field(None, description="Original query string")
    sub_qas: List[QAPair] = Field(None, description="List of question-answer pairs related to the original query")



