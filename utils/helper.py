from .schemas import Reviewer
from typing import List

def print_reviewer(reviewer: Reviewer):
    print(f"ID: {reviewer.id}")
    print(f"Specialisation: {reviewer.specialisation}")
    print(f"Questions:")

    if reviewer.questions is None:
        return
     
    for i, question in enumerate(reviewer.questions):
        print(f"{i+1}. {question}") 

def print_reviewers(reviewers):
    for i, reviewer in enumerate(reviewers):
        print(f"Reviewer {i+1}:")
        print("----")
        print_reviewer(reviewer)