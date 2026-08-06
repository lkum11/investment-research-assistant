from fastapi import FastAPI
from pydantic import BaseModel
from src.graph.graph import graph

app = FastAPI(title="Investment Research Assistant")


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
    result = graph.invoke(initial_state)
    
    return {
        "answer": result["answer"],
        "verdict": result["verdict"],
        "retries": result["retries"],
    }