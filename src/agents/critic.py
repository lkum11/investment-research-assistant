from src.llm import call_llm


def critique(answer: str, chunks: list[str]) -> str:
    """Critic agent: check whether the answer is supported by the chunks."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] {chunk}\n\n"

    prompt = f"""You are a strict fact-checker. Below is an ANSWER and the CONTEXT it was supposed to be based on.

    Your job: check whether every fact and number in the ANSWER actually appears in the CONTEXT.

    Rules:
    - If the answer says it does not have enough information, respond exactly: SUPPORTED
    - Check each specific number and claim in the answer against the context.
    - If every fact is present in the context, respond exactly: SUPPORTED
    - If any fact or number is NOT found in the context, respond exactly: NOT SUPPORTED, then list which facts are missing.

    CONTEXT:
    {context}

    ANSWER:
    {answer}

    Verdict:"""

    return call_llm(prompt)