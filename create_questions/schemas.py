from pydantic import BaseModel, Field
from typing import List, Optional

class Reviewer(BaseModel):
    id: int = Field(description="ID of the reviewer")
    specialisation: str = Field(description="Specialisation of the reviewer")
    questions: List[str] = Field(description="Questions asked by the reviewer")

class QuestionState(BaseModel):
    messages: List[str]
    paper: str
    topic: str
    num_reviewers: int
    reviewers: List[Reviewer]
    conference: str
    conference_description: str



