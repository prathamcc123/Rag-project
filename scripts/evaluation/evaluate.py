"""
Evaluation Script for Agriculture RAG

This script:

1. Loads evaluation questions
2. Runs retrieval
3. Computes Precision@5
4. Computes Recall@5
5. Computes Hit Rate@5
6. Computes Reciprocal Rank
7. Saves results
"""

import csv
import json
from pathlib import Path

from metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    average,
)
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag.retrieve import retrieve_query

# ============================================================
# Configuration
# ============================================================

TEST_FILE = "scripts/evaluation/test_questions.json"

RESULT_FILE = "scripts/evaluation/results.csv"

SUMMARY_FILE = "scripts/evaluation/evaluation_summary.txt"

TOP_K = 5


# ============================================================
# Load Test Questions
# ============================================================

def load_questions():
    """Load evaluation dataset."""

    if not Path(TEST_FILE).exists():
        raise FileNotFoundError(TEST_FILE)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Evaluate One Question
# ============================================================

def evaluate_question(sample):
    """
    Evaluate one test question.
    """

    question = sample["question"]

    expected = sample["expected_article"]

    retrieved_chunks, _ = retrieve_query(
        question,
        verbose=False,
        save_context=False,
    )

    retrieved_titles = [
        chunk["title"]
        for chunk in retrieved_chunks
    ]

    precision = precision_at_k(
        retrieved_titles,
        expected,
        TOP_K,
    )

    recall = recall_at_k(
        retrieved_titles,
        expected,
        TOP_K,
    )

    hit_rate = hit_rate_at_k(
        retrieved_titles,
        expected,
        TOP_K,
    )

    rr = reciprocal_rank(
        retrieved_titles,
        expected,
    )

    return {
        "question": question,
        "expected": expected,
        "retrieved_titles": retrieved_titles,
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "rr": rr,
    }
    # ============================================================
# Save CSV Results
# ============================================================

def save_results(results):
    """
    Save detailed evaluation results to CSV.
    """

    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "Question",
            "Expected Article",
            "Retrieved Titles",
            "Precision@5",
            "Recall@5",
            "HitRate@5",
            "ReciprocalRank",
        ])

        for result in results:

            writer.writerow([
                result["question"],
                result["expected"],
                " | ".join(result["retrieved_titles"]),
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                result["hit_rate"],
                f"{result['rr']:.4f}",
            ])


# ============================================================
# Save Summary
# ============================================================

def save_summary(results):
    """
    Compute overall evaluation metrics and save them.
    """

    precision = average([r["precision"] for r in results])
    recall = average([r["recall"] for r in results])
    hit_rate = average([r["hit_rate"] for r in results])
    mrr = average([r["rr"] for r in results])

    summary = f"""
==========================================================
Agriculture RAG Evaluation Summary
==========================================================

Total Questions : {len(results)}

Precision@5 : {precision:.4f}

Recall@5    : {recall:.4f}

Hit Rate@5  : {hit_rate:.4f}

MRR          : {mrr:.4f}

==========================================================
"""

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)


# ============================================================
# Main
# ============================================================

def main():

    questions = load_questions()

    print("=" * 60)
    print("Running Agriculture RAG Evaluation")
    print("=" * 60)

    results = []

    for i, sample in enumerate(questions, start=1):

        print(f"[{i}/{len(questions)}] {sample['question']}")

        result = evaluate_question(sample)

        results.append(result)

    save_results(results)

    save_summary(results)

    print("✓ Results saved to :", RESULT_FILE)
    print("✓ Summary saved to :", SUMMARY_FILE)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()