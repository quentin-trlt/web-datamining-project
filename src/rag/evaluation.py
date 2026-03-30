"""
Evaluate RAG pipeline: baseline (no KG) vs SPARQL-generation RAG.
Runs a set of predefined questions and compares results.
"""

import csv
import logging
from pathlib import Path

from rdflib import Graph

from rag.rag_engine import answer_no_rag, answer_with_sparql_generation
from utils import project_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Evaluation questions targeting the AI-news KB domain
EVAL_QUESTIONS = [
    "Which organizations develop AI products?",
    "Where is OpenAI headquartered?",
    "Who founded Google?",
    "List all AI products in the knowledge base.",
    "Which companies are located in the United States?",
    "What technologies are used by Microsoft?",
    "Which persons work for AI companies?",
]


def run_evaluation(
    g: Graph,
    schema_summary: str,
    model: str = "gemma:2b",
    output_path: str = None,
) -> list[dict]:
    """
    Run evaluation: for each question, get baseline and RAG answers.
    Returns list of result dicts and saves to CSV.
    """
    if output_path is None:
        output_path = project_path("data/rag_evaluation.csv")
    results = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Question {i}/{len(EVAL_QUESTIONS)}: {question}")
        logger.info(f"{'=' * 60}")

        # Baseline (no RAG)
        logger.info("Getting baseline answer (no KG)...")
        try:
            baseline = answer_no_rag(question, model=model)
        except Exception as e:
            baseline = f"[Error: {e}]"

        # RAG with SPARQL generation
        logger.info("Getting RAG answer (SPARQL generation)...")
        try:
            rag_result = answer_with_sparql_generation(
                g, schema_summary, question, model=model
            )
        except Exception as e:
            rag_result = {
                "query": "",
                "vars": [],
                "rows": [],
                "repaired": False,
                "attempts": 0,
                "error": str(e),
            }

        # Format RAG results
        rag_rows = rag_result.get("rows", [])
        if rag_rows:
            vars_ = rag_result.get("vars", [])
            rag_answer = "; ".join(
                [", ".join(r) for r in rag_rows[:10]]
            )
            if len(rag_rows) > 10:
                rag_answer += f" ... ({len(rag_rows)} total)"
        else:
            rag_answer = rag_result.get("error", "No results")

        result = {
            "question": question,
            "baseline_answer": baseline[:500],  # truncate long answers
            "rag_sparql": rag_result.get("query", ""),
            "rag_result": rag_answer[:500],
            "rag_repaired": rag_result.get("repaired", False),
            "rag_attempts": rag_result.get("attempts", 0),
            "rag_error": rag_result.get("error", ""),
            "rag_num_rows": len(rag_rows),
        }
        results.append(result)

        # Display
        logger.info(f"\n[Baseline]: {baseline[:200]}")
        logger.info(f"[RAG SPARQL]: {rag_result.get('query', 'N/A')}")
        logger.info(f"[RAG Result]: {rag_answer[:200]}")
        logger.info(f"[Repaired]: {rag_result.get('repaired', False)} "
                     f"(attempts: {rag_result.get('attempts', 0)})")

    # Save to CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nEvaluation results saved to {out}")
    return results


def print_evaluation_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print("EVALUATION: Baseline (No RAG) vs SPARQL-Generation RAG")
    print(f"{'=' * 80}")

    for i, r in enumerate(results, 1):
        print(f"\n--- Question {i}: {r['question']}")
        print(f"  Baseline : {r['baseline_answer'][:120]}...")
        print(f"  RAG SPARQL: {r['rag_sparql'][:120]}")
        print(f"  RAG Result: {r['rag_result'][:120]}")
        print(f"  Repaired: {r['rag_repaired']} | Rows: {r['rag_num_rows']} | "
              f"Attempts: {r['rag_attempts']}")

    print(f"\n{'=' * 80}")
    total = len(results)
    with_results = sum(1 for r in results if r["rag_num_rows"] > 0)
    repaired = sum(1 for r in results if r["rag_repaired"])
    print(f"Summary: {with_results}/{total} questions returned results, "
          f"{repaired} required self-repair")
