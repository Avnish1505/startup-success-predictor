"""Measure recall@3 and recall@5 of the real built index against the hand-
written eval set, and write RETRIEVAL_EVAL.md with the real numbers."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.advisor.retrieval import load_index, retrieve  # noqa: E402

INDEX_DIR = REPO_ROOT / "models" / "production" / "advisor_index"
EVAL_PATH = REPO_ROOT / "data" / "corpus" / "eval_questions.json"
OUT_PATH = REPO_ROOT / "RETRIEVAL_EVAL.md"


def recall_at_k(index, eval_cases: list[dict], k: int) -> float:
    hits = 0
    scored_cases = [c for c in eval_cases if c["expected_chunk_ids"]]
    for case in scored_cases:
        results = retrieve(case["question"], index, top_k=k)
        retrieved_ids = {chunk.chunk_id for chunk, _ in results}
        if retrieved_ids & set(case["expected_chunk_ids"]):
            hits += 1
    return hits / len(scored_cases) if scored_cases else 0.0


def main() -> None:
    index = load_index(INDEX_DIR)
    eval_cases = json.loads(EVAL_PATH.read_text())

    recall_3 = recall_at_k(index, eval_cases, k=3)
    recall_5 = recall_at_k(index, eval_cases, k=5)
    n_scored = len([c for c in eval_cases if c["expected_chunk_ids"]])
    n_out_of_corpus = len(eval_cases) - n_scored

    print(f"recall@3: {recall_3:.3f}")
    print(f"recall@5: {recall_5:.3f}")
    print(f"n questions: {len(eval_cases)} ({n_scored} scored, {n_out_of_corpus} deliberately out-of-corpus)")

    lines = [
        "# Retrieval Evaluation",
        "",
        f"Measured against {len(eval_cases)} hand-written questions ({n_scored} with an expected chunk, "
        f"{n_out_of_corpus} deliberately out-of-corpus) over the real built index "
        f"(`models/production/advisor_index/`, {len(index.chunks)} chunks from `data/corpus/*.md`).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| recall@3 | {recall_3:.3f} |",
        f"| recall@5 | {recall_5:.3f} |",
        "",
        "recall@k = fraction of questions with a known expected chunk where at least one expected "
        "chunk ID appeared in the top-k retrieved results (before MMR reranking - MMR reorders/limits "
        "the final k shown to the user, but retrieval quality itself is what this measures).",
        "",
        "Run `python3 scripts/run_retrieval_eval.py` to reproduce - these numbers are generated, "
        "not hand-typed.",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
