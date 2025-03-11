from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

load_dotenv()

class State(BaseModel):
    pass


graph_builder = StateGraph(State)
graph_builder.compile().get_graph().draw_mermaid_png()
