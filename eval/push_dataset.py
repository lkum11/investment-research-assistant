import src.config
from langfuse import get_client
import json

langfuse = get_client()

with open("eval/golden_set.json") as f:
    golden_set = json.load(f)

DATASET_NAME = "apple-earnings-qa"

langfuse.create_dataset(name=DATASET_NAME)

for item in golden_set:
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        id=str(item["id"]),
        input={"question": item["question"]},
        expected_output=item["ground_truth"],
    )

print(f"Pushed {len(golden_set)} items to dataset '{DATASET_NAME}'")