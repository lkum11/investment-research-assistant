import os
from dotenv import load_dotenv

# Load .env once, here, so config is the single source of truth
load_dotenv()

# Model is read from .env if present, otherwise falls back to this default
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# The API key also lives as config, read from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")