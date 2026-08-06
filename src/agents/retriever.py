from src.retrieval.vector_store import search

def retrieve(question: str, k: int = 3) -> list[str]:
    """Retriever agent: find the chunks most relevant to the question."""
    return search(query=question, k=k)