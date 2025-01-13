from schemas import Reviewer
from typing import List

def print_reviewer(reviewer: Reviewer):
    print(f"ID: {reviewer.id}")
    print(f"Specialisation: {reviewer.specialisation}")
    print(f"Questions:\n")
    for i, question in reviewer.questions:
        print(f"{i+1}. {question}\n") 
    print("\n")

def print_reviewers(reviewers: List[Reviewer]):
    for i, reviewer in enumerate(reviewers):
        print(f"Reviewer {i+1}:\n")
        print("----")
        print_reviewer(reviewer)
        print("\n")