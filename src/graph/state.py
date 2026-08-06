from typing import TypedDict

class ResearchState(TypedDict):
    question: str
    chunks: list[str]
    answer: str
    verdict: str
    retries: int