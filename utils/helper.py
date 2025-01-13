from .schemas import Reviewer
from typing import List
import subprocess
import signal
import os

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

def kill_process_on_port(port):
    current_pid = str(os.getpid())
    result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')

    for pid in pids:
        if pid and pid != current_pid:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process {pid} on port {port}")
        else:
            print(f"Skipped killing current process {pid}")
