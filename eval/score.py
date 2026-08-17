import json
from src.llm import call_llm

with open("eval/results.json") as f:
    results = json.load(f)


def judge(question, answer, ground_truth, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are a strict evaluator of a financial QA system. Grade the ANSWER on two things.

    QUESTION: {question}

    GROUND TRUTH (the correct answer): {ground_truth}

    RETRIEVED CONTEXT (what the system was given to work from):
    {context_text}

    SYSTEM ANSWER: {answer}

    Grade two things, each YES or NO:
    1. CORRECT: Does the system answer match the ground truth?
        - If the ground truth is a real value and the system refused or said it lacks info, that is NO (wrong).
        - A refusal is only CORRECT if the ground truth itself says the information is unavailable.
    2. FAITHFUL: Is every number/fact in the system answer actually present in the retrieved context? (If the answer is a refusal, answer YES.)

    Respond in EXACTLY this format:
    CORRECT: YES or NO
    FAITHFUL: YES or NO"""

    return call_llm(prompt)


correct_count = 0
faithful_count = 0

for r in results:
    verdict = judge(r["question"], r["answer"], r["ground_truth"], r["contexts"])
    is_correct = "CORRECT: YES" in verdict
    is_faithful = "FAITHFUL: YES" in verdict
    correct_count += is_correct
    faithful_count += is_faithful
    print(f"Q{r['id']}: correct={is_correct}, faithful={is_faithful}")

total = len(results)
print("\n--- SCORES ---")
print(f"Correctness: {correct_count}/{total} = {correct_count/total:.0%}")
print(f"Faithfulness: {faithful_count}/{total} = {faithful_count/total:.0%}")