import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rag_demo import build_rag_system, gemini_answer, load_config, make_prompt, retrieve

EVAL_FILE = Path("examples/eval_questions.jsonl")
OUTPUT_FILE = Path("rag_eval_results.csv")


def load_eval_questions(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def contains_any(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def first_relevant_rank(retrieved, keywords):
    for rank, chunk in enumerate(retrieved, start=1):
        if contains_any(chunk["text"], keywords):
            return rank
    return None


def main():
    load_dotenv()
    cfg = load_config()
    chunks, index, embedder = build_rag_system(cfg)

    rows = []
    for ex in load_eval_questions(EVAL_FILE):
        question = ex["question"]
        print(f"Evaluating: {question}")

        retrieved = retrieve(question, index, chunks, embedder, cfg["top_k"])
        prompt = make_prompt(question, retrieved)
        answer = gemini_answer(prompt, cfg["llm_model"], retrieved=retrieved)

        retrieved_text = "\n\n".join(c["text"] for c in retrieved)
        rows.append(
            {
                "id": ex["id"],
                "question": question,
                "retrieval_hit": contains_any(retrieved_text, ex["expected_context_keywords"]),
                "first_relevant_rank": first_relevant_rank(retrieved, ex["expected_context_keywords"]),
                "answer_keyword_hit": contains_any(answer, ex["expected_answer_keywords"]),
                "answer": answer,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEvaluation summary")
    print("------------------")
    print(f"Questions: {len(df)}")
    print(f"Retrieval hit rate: {df['retrieval_hit'].mean():.3f}")
    print(f"Answer keyword hit rate: {df['answer_keyword_hit'].mean():.3f}")
    print(f"Saved detailed results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
