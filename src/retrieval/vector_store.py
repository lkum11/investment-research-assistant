import chromadb
from openai import OpenAI
from src.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma client that saves to a local folder on disk
chroma_client = chromadb.PersistentClient(path="data/chroma")
collection = chroma_client.get_or_create_collection("earnings")


def embed(text: str) -> list[float]:
    """Turn one piece of text into a vector using OpenAI embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def add_chunks(chunks: list[str], source: str):
    """Embed each chunk and store it in Chroma with its source filename."""
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{source}_{i}"],
            embeddings=[embed(chunk)],
            documents=[chunk],
            metadatas=[{"source": source}],
        )


def search(query: str, k: int = 3) -> list[str]:
    """Embed the question, return the k most similar chunks."""
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=k,
    )
    return results["documents"][0]