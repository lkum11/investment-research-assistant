from src.llm import call_llm

def synthesize(question: str, chunks: list[str]) -> str:
    """Synthesizer agent: write a grounded answer from the retrieved chunks."""
    # Number the chunks so the model can cite them as [1], [2], ...
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] {chunk} \n\n"

    prompt = f"""You are a financial research assistant. Answer the question using ONLY the context below.

    Rules:
    - Use only facts found in the context. Do not use outside knowledge.
    - The context may contain financial tables where labels and numbers are separated by spacing. Read them carefully and match labels to their values.
    - Cite the chunk number after each fact, like [1] or [2].
    - If the context does not contain the answer, say exactly: "The provided documents do not contain enough information to answer this."

    Context:
    {context}

    Question: {question}

    Answer:"""

    return call_llm(prompt=prompt)