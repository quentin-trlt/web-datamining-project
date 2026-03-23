"""
Knowledge Graph construction pipeline orchestrator.
Runs: ontology → build KB → entity linking → predicate alignment → expansion.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="KB Construction Pipeline")
    parser.add_argument("--skip-ontology", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-linking", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-expansion", action="store_true")
    parser.add_argument("--entities-csv", default="data/extracted_entities.csv")
    parser.add_argument("--relations-csv", default="data/extracted_relations.csv")
    parser.add_argument("--max-triples", type=int, default=150000)
    args = parser.parse_args()

    # Step 1: Build ontology
    if not args.skip_ontology:
        logger.info("=" * 60)
        logger.info("Step 1: Building ontology")
        from kg.ontology import run_ontology
        run_ontology()

    # Step 2: Build initial KB from Lab 1 CSVs
    if not args.skip_build:
        logger.info("=" * 60)
        logger.info("Step 2: Building initial Knowledge Base")
        from kg.build_kb import build_kb
        build_kb(
            entities_csv=args.entities_csv,
            relations_csv=args.relations_csv,
        )

    # Step 3: Entity linking with Wikidata
    if not args.skip_linking:
        logger.info("=" * 60)
        logger.info("Step 3: Entity linking with Wikidata")
        from kg.entity_linking import run_entity_linking
        run_entity_linking(entities_csv=args.entities_csv)

    # Step 4: Predicate alignment
    if not args.skip_alignment:
        logger.info("=" * 60)
        logger.info("Step 4: Predicate alignment via SPARQL")
        from kg.predicate_alignment import run_predicate_alignment
        run_predicate_alignment()

    # Step 5: KB expansion
    if not args.skip_expansion:
        logger.info("=" * 60)
        logger.info("Step 5: KB expansion via Wikidata SPARQL")
        from kg.expand_kb import run_expansion
        run_expansion(max_triples=args.max_triples)

    logger.info("=" * 60)
    logger.info("KB construction pipeline complete!")


if __name__ == "__main__":
    main()
