"""
Data preparation for Knowledge Graph Embedding:
Load expanded KB, clean triples, split into train/valid/test.
"""

import json
import logging
import random
from pathlib import Path

from rdflib import Graph, URIRef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_expanded_kb(nt_path: str) -> list[tuple[str, str, str]]:
    """Parse N-Triples file and return list of (head, relation, tail) string tuples."""
    g = Graph()
    path = Path(nt_path)

    if path.suffix == ".nt":
        g.parse(str(path), format="nt")
    elif path.suffix == ".ttl":
        g.parse(str(path), format="turtle")
    else:
        g.parse(str(path))

    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))

    logger.info(f"Loaded {len(triples)} entity-to-entity triples from {nt_path}")
    return triples


def clean_triples(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """
    Clean triples for embedding:
    - Remove duplicates
    - Remove self-loops (head == tail)
    - Remove RDF/OWL schema triples that aren't useful for embedding
    """
    # Schema predicates to exclude
    schema_predicates = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
        "http://www.w3.org/2000/01/rdf-schema#domain",
        "http://www.w3.org/2000/01/rdf-schema#range",
        "http://www.w3.org/2002/07/owl#sameAs",
        "http://www.w3.org/2002/07/owl#equivalentProperty",
        "http://www.w3.org/2002/07/owl#equivalentClass",
        "http://purl.org/dc/terms/source",
    }

    seen = set()
    cleaned = []
    removed_dup = 0
    removed_self = 0
    removed_schema = 0

    for h, r, t in triples:
        # Skip schema triples
        if r in schema_predicates:
            removed_schema += 1
            continue

        # Skip self-loops
        if h == t:
            removed_self += 1
            continue

        # Skip duplicates
        triple = (h, r, t)
        if triple in seen:
            removed_dup += 1
            continue

        seen.add(triple)
        cleaned.append(triple)

    logger.info(
        f"Cleaning: {len(triples)} → {len(cleaned)} triples "
        f"(removed {removed_dup} duplicates, {removed_self} self-loops, {removed_schema} schema)"
    )
    return cleaned


def build_indices(triples: list[tuple[str, str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """Build entity2id and relation2id mappings."""
    entities = set()
    relations = set()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    logger.info(f"Indices: {len(entity2id)} entities, {len(relation2id)} relations")
    return entity2id, relation2id


def split_dataset(
    triples: list[tuple[str, str, str]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[list, list, list]:
    """
    Split triples into train/valid/test ensuring no entity leakage.
    Entities in valid/test must also appear in at least one training triple.
    """
    random.seed(RANDOM_SEED)
    shuffled = list(triples)
    random.shuffle(shuffled)

    # First pass: assign all triples to train
    train = list(shuffled)
    valid = []
    test = []

    # Build entity set from all triples
    train_entities = set()
    for h, r, t in train:
        train_entities.add(h)
        train_entities.add(t)

    # Move triples to valid/test only if both entities appear in remaining train
    target_valid = int(len(shuffled) * valid_ratio)
    target_test = int(len(shuffled) * (1 - train_ratio - valid_ratio))

    # Build entity frequency in train
    entity_count = {}
    for h, r, t in train:
        entity_count[h] = entity_count.get(h, 0) + 1
        entity_count[t] = entity_count.get(t, 0) + 1

    # Move triples to valid set
    new_train = []
    for h, r, t in train:
        if (
            len(valid) < target_valid
            and entity_count.get(h, 0) > 1
            and entity_count.get(t, 0) > 1
        ):
            valid.append((h, r, t))
            entity_count[h] -= 1
            entity_count[t] -= 1
        elif (
            len(test) < target_test
            and entity_count.get(h, 0) > 1
            and entity_count.get(t, 0) > 1
        ):
            test.append((h, r, t))
            entity_count[h] -= 1
            entity_count[t] -= 1
        else:
            new_train.append((h, r, t))

    train = new_train

    logger.info(f"Split: train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test


def save_splits(
    train: list[tuple],
    valid: list[tuple],
    test: list[tuple],
    output_dir: str = "kge_datasets",
) -> None:
    """Save splits as tab-separated files (head \\t relation \\t tail)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.txt", train), ("valid.txt", valid), ("test.txt", test)]:
        filepath = out / name
        with open(filepath, "w") as f:
            for h, r, t in data:
                f.write(f"{h}\t{r}\t{t}\n")
        logger.info(f"Saved {filepath}: {len(data)} triples")


def run_data_preparation(
    expanded_nt: str = "kg_artifacts/expanded.nt",
    output_dir: str = "kge_datasets",
    stats_output: str = "data/kge_data_stats.json",
) -> dict:
    """Run the full data preparation pipeline."""
    # Load
    triples = load_expanded_kb(expanded_nt)

    # Clean
    triples = clean_triples(triples)

    # Build indices
    entity2id, relation2id = build_indices(triples)

    # Split
    train, valid, test = split_dataset(triples)

    # Save splits
    save_splits(train, valid, test, output_dir)

    # Save indices
    indices_dir = Path(output_dir)
    with open(indices_dir / "entity2id.json", "w") as f:
        json.dump(entity2id, f)
    with open(indices_dir / "relation2id.json", "w") as f:
        json.dump(relation2id, f)

    # Statistics
    stats = {
        "total_triples": len(triples),
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "train_size": len(train),
        "valid_size": len(valid),
        "test_size": len(test),
    }

    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Data preparation complete. Stats: {stats}")
    return stats


if __name__ == "__main__":
    run_data_preparation()
