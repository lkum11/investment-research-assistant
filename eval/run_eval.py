import json
from src.graph.graph import graph
from src.agents.retriever import retrieve
from src.config import CHUNK_STRATEGY


# Load the golden set
with open("eval/golden_set.json") as f:
    golden_set = json.load(f)

results = []

for item in golden_set:
    question = item["question"]
    print(f"Running Q{item['id']}: {question[:60]}...")

    # Grab the chunks separately (RAGAS needs them for faithfulness)
    chunks = retrieve(question=question)

    # Run the full graph to get the answer
    final_state = graph.invoke({
        "question": question,
        "chunks": [],
        "answer": "",
        "verdict": "",
        "retries" : 0
    })

    results.append({
        "id": item["id"],
        "question": question,
        "ground_truth": item["ground_truth"],
        "answer": final_state["answer"],
        "contexts": chunks,# the retrieved chunks
        "verdict": final_state["verdict"],
    })

    # Save everything
    output_path = f"eval/results_{CHUNK_STRATEGY}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Ran {len(results)} questions. Saved to {output_path}")
    # # Save everything
    # with open("eval/results.json", "w") as f:
    #     json.dump(results, f, indent=2)

    print(f"\nDone. Ran {len(results)} questions. Saved to eval/results.json")