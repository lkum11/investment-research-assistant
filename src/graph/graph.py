from langgraph.graph import StateGraph, END
from src.graph.state import ResearchState
from src.graph.nodes import (
    retrieve_node,
    synthesize_node,
    critique_node,
    route_after_critique
)

# Build the graph over our state shape
builder = StateGraph(ResearchState)

builder.add_node("retrieve",retrieve_node)
builder.add_node("synthesize",synthesize_node)
builder.add_node("critique",critique_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "synthesize")
builder.add_edge("synthesize", "critique")

builder.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "retry": "synthesize",
        "end": END
    }
)

# Compile into a runnable graph
graph = builder.compile()
