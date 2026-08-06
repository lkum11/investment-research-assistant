from src.agents.retriever import retrieve
from src.agents.synthesizer import synthesize
from src.agents.critic import critique
from src.graph.state import ResearchState


# LangGraph MERGES a node's returned dict into the shared state (patch, not replace).
# A node returns ONLY the fields it changed; LangGraph patches those keys into
# ResearchState and leaves all other fields untouched. The dict keys must match
# the field names in ResearchState (e.g. "chunks", "answer").
# So {"chunks": chunks} updates state["chunks"] and nothing else.
def retrieve_node(state: ResearchState) -> dict:
    chunks = retrieve(state["question"])
    return {"chunks": chunks}


def synthesize_node(state: ResearchState) -> dict:
    answer = synthesize(state["question"], state["chunks"])
    return { "answer": answer}

def critique_node(state: ResearchState) -> dict:
    verdict = critique(state["answer"], state["chunks"])
    return { "verdict": verdict, "retries": state["retries"] + 1}


def route_after_critique(state: ResearchState) -> str:
    """Decide what happens after the critic runs."""
    if state["verdict"].startswith("SUPPORTED") or state["retries"] >= 2:
        return "end"
    return "retry"