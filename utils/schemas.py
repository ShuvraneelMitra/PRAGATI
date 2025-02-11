from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from typing_extensions import TypedDict

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class Reviewer(BaseModel):
    id: int = Field(None, description="ID of the reviewer")
    specialisation: str = Field(None, description="Specialisation of the reviewer")
    questions: List[str] = Field(None, description="Questions asked by the reviewer")
    conference: str = Field(None, description="Conference of the reviewer")
    conference_description: str = Field(None, description="Description of the conference")
    topic: str = Field(None, description="Topic of the paper being reviewed")

class ReviewerProps(BaseModel):
    specialisation: str = Field(None, description="Specialisation Field of the reviewer. This must be aligned with the conference topic")

class QAPair(BaseModel):
    query: str = Field(None, description="Query string")
    answer: str = Field(None, description="Answer string")
    references: List[str] = Field(None, description="List of references related to the query and answer")

class Queries(BaseModel):
    original_query: str = Field(None, description="Original query string")
    sub_qas: List[QAPair] = Field(None, description="List of question-answer pairs related to the original query")

class Paper(BaseModel):
    object_id: str = Field(None, description="Object ID of the paper")
    title: str = Field(None, description="Title of the paper")
    filename: str = Field(None, description="Filename of the paper")



