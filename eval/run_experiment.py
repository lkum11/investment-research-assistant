import src.config  # loads .env
from langfuse import get_client
from src.graph.graph import graph
from src.agents.retriever import retrieve

langfuse = get_client()


def my_task(*, item, **kwargs):
    """Runs your pipeline on one dataset item, returns the answer."""
    question = item.input["question"]
    final_state = graph.invoke({
        "question": question,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "retries": 0,
    })
    return final_state["answer"]


dataset = langfuse.get_dataset("apple-earnings-qa")

result = dataset.run_experiment(
    name="fixed-chunking-v1",
    description="Baseline: fixed-size chunking",
    task=my_task,
)

print(result.format())