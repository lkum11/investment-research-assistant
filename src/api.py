from fastapi import FastAPI
from pydantic import BaseModel
from langfuse.langchain import CallbackHandler
from src.graph.graph import graph

app = FastAPI(title="Investment Research Assistant")
lanfuse_handler = CallbackHandler()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(request: Question):

    initial_state = {
        "question": request.question,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "retries": 0,
    }
    result = graph.invoke(
        initial_state,
        config={"callbacks": [lanfuse_handler]}
    )

    return {
        "answer": result["answer"],
        "verdict": result["verdict"],
        "retries": result["retries"],
    }