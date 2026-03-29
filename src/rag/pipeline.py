"""
RAG pipeline orchestrator: CLI demo with interactive REPL and batch evaluation.
"""

import argparse
import logging

from rag.rag_engine import (
    answer_no_rag,
    answer_with_sparql_generation,
    build_schema_summary,
    load_graph,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pretty_print_result(result: dict) -> None:
    """Pretty-print a RAG result."""
    if result.get("error"):
        print(f"\n[Execution Error] {result['error']}")

    print(f"\n[SPARQL Query Used]")
    print(result["query"])
    print(f"\n[Repaired?] {result['repaired']} (attempts: {result['attempts']})")

    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("\n[No rows returned]")
        return

    print(f"\n[Results] ({len(rows)} rows)")
    print("  " + " | ".join(vars_))
    print("  " + "-" * (sum(len(v) for v in vars_) + 3 * len(vars_)))
    for r in rows[:20]:
        print("  " + " | ".join(r))
    if len(rows) > 20:
        print(f"  ... (showing 20 of {len(rows)})")


def interactive_mode(g, schema_summary: str, model: str) -> None:
    """Interactive REPL: ask questions, compare baseline vs RAG."""
    print("\n" + "=" * 60)
    print("RAG Demo — NL to SPARQL over AI-News Knowledge Graph")
    print(f"Model: {model}")
    print(f"Graph: {len(g)} triples loaded")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not q:
            continue

        # Baseline
        print("\n--- Baseline (No RAG) ---")
        try:
            baseline = answer_no_rag(q, model=model)
            print(baseline)
        except Exception as e:
            print(f"[Error] {e}")

        # RAG
        print(f"\n--- SPARQL-generation RAG ({model} + rdflib) ---")
        try:
            result = answer_with_sparql_generation(
                g, schema_summary, q, model=model
            )
            pretty_print_result(result)
        except Exception as e:
            print(f"[Error] {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline — NL to SPARQL")
    parser.add_argument(
        "--kg-path",
        default="kg_artifacts/expanded.nt",
        help="Path to the RDF knowledge graph (default: kg_artifacts/expanded.nt)",
    )
    parser.add_argument(
        "--model",
        default="gemma:2b",
        help="Ollama model name (default: gemma:2b)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run batch evaluation instead of interactive mode",
    )
    args = parser.parse_args()

    # Load graph and build schema
    logger.info("Loading knowledge graph...")
    g = load_graph(args.kg_path)

    logger.info("Building schema summary...")
    schema_summary = build_schema_summary(g)
    logger.info(f"Schema summary built ({len(schema_summary)} chars)")

    if args.evaluate:
        # Batch evaluation mode
        from rag.evaluation import print_evaluation_table, run_evaluation

        results = run_evaluation(g, schema_summary, model=args.model)
        print_evaluation_table(results)
    else:
        # Interactive mode
        interactive_mode(g, schema_summary, args.model)


if __name__ == "__main__":
    main()
