"""
Ingest earnings PDFs into the vector store.

Chunking strategy is selected by the CHUNK_STRATEGY environment variable
("fixed" or "recursive"), which also determines the target Chroma collection
(earnings_fixed / earnings_recursive). This lets both strategies coexist so
they can be compared without re-embedding.

USAGE — build both collections (run once each, from the project root):

    CHUNK_STRATEGY=fixed     python -m ingest
    CHUNK_STRATEGY=recursive python -m ingest

Re-running the same strategy is safe: chunks are keyed by source + index, so
re-ingestion overwrites rather than duplicates.

Prerequisites: venv active, dependencies installed (pip install -r
requirements.txt), and OPENAI_API_KEY set in .env.
"""

import src.config  # loads .env
from src.loaders.pdf_loader import load_pdf
from src.chunking import chunk_text            # swappable: uses CHUNK_STRATEGY
from src.retrieval.vector_store import add_chunks
from src.config import CHUNK_STRATEGY, COLLECTION_NAME

PDF_PATH = "data/raw/apple_earnings_2026q3.pdf"
SOURCE = "apple_earnings_2026q3.pdf"

print(f"Ingesting with strategy='{CHUNK_STRATEGY}' into collection='{COLLECTION_NAME}'")

text = load_pdf(PDF_PATH)
chunks = chunk_text(text)
add_chunks(chunks, source=SOURCE)

print(f"Done. Ingested {len(chunks)} chunks.")