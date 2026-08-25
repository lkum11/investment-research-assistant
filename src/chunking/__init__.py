from src.config import CHUNK_STRATEGY
from src.chunking import fixed_chunker, recursive_chunker

_STRATEGIES = {
    "fixed": fixed_chunker.chunk_text,
    "recursive": recursive_chunker.chunk_text,
}


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Dispatch to the chunker chosen by CHUNK_STRATEGY."""
    return _STRATEGIES[CHUNK_STRATEGY](text, chunk_size, overlap)