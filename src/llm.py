# from openai import OpenAI
from langfuse.openai import OpenAI
from src.config import LLM_MODEL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def call_llm(prompt: str) -> str:
    """The ONE place the whole app talks to an LLM.
    Model and key come from config; we can swap internals here later (e.g. Bedrock)."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
