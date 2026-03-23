"""
SWRL reasoning on family.owl: infer OldPerson class for persons older than 60.
Uses OWLReady2 with Pellet reasoner.
"""

import logging
from pathlib import Path

import owlready2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_family_ontology(owl_path: str = "kg_artifacts/family.owl") -> owlready2.Ontology:
    """Load the family ontology from an OWL file."""
    path = Path(owl_path).resolve()
    onto = owlready2.get_ontology(path.as_uri()).load()
    logger.info(f"Loaded ontology: {onto.base_iri}")
    logger.info(f"  Classes: {list(onto.classes())}")
    logger.info(f"  Individuals: {list(onto.individuals())}")
    return onto


def add_old_person_rule(onto: owlready2.Ontology) -> None:
    """
    Add SWRL rule: Person(?p) ∧ hasAge(?p, ?age) ∧ swrlb:greaterThan(?age, 60) → OldPerson(?p)
    """
    with onto:
        rule = owlready2.Imp()
        rule.set_as_rule(
            "family:Person(?p), family:hasAge(?p, ?age), greaterThan(?age, 60) -> family:OldPerson(?p)"
        )
    logger.info("SWRL rule added: Person(?p) ∧ hasAge(?p, ?age) ∧ greaterThan(?age, 60) → OldPerson(?p)")


def run_reasoner_and_display(onto: owlready2.Ontology) -> list[str]:
    """Run the Pellet reasoner and display inferred OldPerson instances."""
    logger.info("Running reasoner (Pellet)...")
    try:
        owlready2.sync_reasoner_pellet(
            infer_property_values=True,
            infer_data_property_values=True,
        )
    except Exception as e:
        logger.warning(f"Pellet failed, trying HermiT: {e}")
        owlready2.sync_reasoner()

    # Query OldPerson instances
    old_person_class = onto["OldPerson"]
    if old_person_class is None:
        logger.error("OldPerson class not found in ontology")
        return []

    inferred = list(old_person_class.instances())

    logger.info(f"\n{'=' * 50}")
    logger.info(f"SWRL Reasoning Results — OldPerson (age > 60)")
    logger.info(f"{'=' * 50}")

    results = []
    for ind in inferred:
        name = ind.name
        age = ind.hasAge[0] if hasattr(ind, "hasAge") and ind.hasAge else "?"
        logger.info(f"  {name} (age: {age}) → OldPerson ✓")
        results.append(name)

    logger.info(f"\nTotal inferred OldPerson instances: {len(results)}")

    # Also show non-old persons
    person_class = onto["Person"]
    if person_class:
        all_persons = list(person_class.instances())
        non_old = [p for p in all_persons if p not in inferred]
        logger.info(f"\nPersons NOT classified as OldPerson:")
        for p in non_old:
            age = p.hasAge[0] if hasattr(p, "hasAge") and p.hasAge else "?"
            logger.info(f"  {p.name} (age: {age})")

    return results


def run_family_swrl(owl_path: str = "kg_artifacts/family.owl") -> list[str]:
    """Run the complete family SWRL demonstration."""
    logger.info("=" * 60)
    logger.info("Part 1: SWRL Reasoning on family.owl")
    logger.info("=" * 60)

    onto = load_family_ontology(owl_path)
    add_old_person_rule(onto)
    results = run_reasoner_and_display(onto)
    return results


if __name__ == "__main__":
    run_family_swrl()
