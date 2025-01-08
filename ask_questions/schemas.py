from pydantic import BaseModel, Field
from typing import List

class Reviewer(BaseModel):
    name: str = Field(description="Name of the reviewer")
    specialisation: str = Field(description="Specialisation of the reviewer")
    questions: List[str] = Field(description="Questions asked by the reviewer")

class QuestionState(BaseModel):
    messages: List[str]
    reviewers: List[Reviewer]



