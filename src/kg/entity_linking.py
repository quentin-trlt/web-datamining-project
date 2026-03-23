"""
Entity linking: match private KB entities to Wikidata using the wbsearchentities API.
Produces alignment triples (owl:sameAs) and a mapping table.
"""

import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import pandas as pd
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB = Namespace("http://example.org/ai-news/")
WD = Namespace("http://www.wikidata.org/entity/")

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Map spaCy labels to Wikidata type filters for better matching
LABEL_TYPE_HINTS = {
    "PERSON": "Q5",        # human
    "ORG": "Q43229",       # organization
    "GPE": "Q515",         # city (broad)
    "PRODUCT": "Q2424752", # product
    "EVENT": "Q1656682",   # event
}


def _query_wikidata_search(entity_name: str, language: str = "en", limit: int = 5) -> list[dict]:
    """Search Wikidata for entities matching the given name."""
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": language,
        "limit": limit,
        "format": "json",
    }
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(WIKIDATA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("search", [])
    except Exception as e:
        logger.warning(f"Wikidata search failed for '{entity_name}': {e}")
        return []


def _compute_confidence(entity_name: str, candidate: dict) -> float:
    """Compute string similarity between entity name and Wikidata candidate."""
    candidate_label = candidate.get("label", "")
    return SequenceMatcher(None, entity_name.lower(), candidate_label.lower()).ratio()


def link_entities(
    graph: Graph,
    entities_df: pd.DataFrame,
    confidence_threshold: float = 0.7,
) -> tuple[Graph, pd.DataFrame]:
    """
    Link entities from the KB to Wikidata.
    Returns an alignment graph and a mapping table DataFrame.
    """
    alignment_graph = Graph()
    alignment_graph.bind("aib", AIB)
    alignment_graph.bind("wd", WD)
    alignment_graph.bind("owl", OWL)

    mapping_rows = []

    # Get unique entities
    unique_entities = entities_df.drop_duplicates(subset=["entity", "label"])[["entity", "label"]]
    total = len(unique_entities)

    for idx, (_, row) in enumerate(unique_entities.iterrows()):
        name = str(row["entity"])
        label = str(row["label"])

        logger.info(f"[{idx + 1}/{total}] Linking: {name} ({label})")

        candidates = _query_wikidata_search(name)
        time.sleep(0.5)  # rate limiting

        best_match = None
        best_confidence = 0.0

        for cand in candidates:
            conf = _compute_confidence(name, cand)
            if conf > best_confidence:
                best_confidence = conf
                best_match = cand

        from kg.build_kb import _normalize_uri
        slug = _normalize_uri(name)
        if not slug:
            continue
        entity_uri = AIB[slug]

        if best_match and best_confidence >= confidence_threshold:
            wd_id = best_match["id"]
            wd_uri = WD[wd_id]
            alignment_graph.add((entity_uri, OWL.sameAs, wd_uri))
            alignment_graph.add((entity_uri, RDFS.label, Literal(name)))

            mapping_rows.append({
                "entity": name,
                "spacy_label": label,
                "wikidata_id": wd_id,
                "wikidata_label": best_match.get("label", ""),
                "wikidata_description": best_match.get("description", ""),
                "confidence": round(best_confidence, 3),
                "matched": True,
            })
            logger.info(f"  Matched: {wd_id} ({best_match.get('label', '')}) conf={best_confidence:.2f}")
        else:
            # Define locally in ontology
            _define_local_entity(alignment_graph, entity_uri, name, label)
            mapping_rows.append({
                "entity": name,
                "spacy_label": label,
                "wikidata_id": "",
                "wikidata_label": "",
                "wikidata_description": "",
                "confidence": round(best_confidence, 3) if best_match else 0.0,
                "matched": False,
            })
            logger.info(f"  No match (best conf={best_confidence:.2f})")

    mapping_df = pd.DataFrame(mapping_rows)
    logger.info(
        f"Entity linking complete: {mapping_df['matched'].sum()}/{total} matched"
    )
    return alignment_graph, mapping_df


def _define_local_entity(graph: Graph, uri: URIRef, name: str, spacy_label: str) -> None:
    """Define a local entity that has no Wikidata match."""
    label_to_class = {
        "PERSON": AIB.Person,
        "ORG": AIB.Organization,
        "GPE": AIB.Location,
        "PRODUCT": AIB.Product,
        "EVENT": AIB.Event,
        "NORP": AIB.NationalGroup,
    }
    rdf_class = label_to_class.get(spacy_label, AIB.Thing)
    graph.add((uri, RDF.type, rdf_class))
    graph.add((uri, RDFS.label, Literal(name)))
    graph.add((uri, RDFS.comment, Literal(f"Local entity (no Wikidata match) — {spacy_label}")))


def run_entity_linking(
    initial_kb_path: str = "kg_artifacts/initial_kb.ttl",
    entities_csv: str = "data/extracted_entities.csv",
    alignment_output: str = "kg_artifacts/alignment.ttl",
    mapping_output: str = "data/entity_mapping.csv",
) -> tuple[Path, Path]:
    """Run the full entity linking pipeline."""
    graph = Graph()
    graph.parse(initial_kb_path, format="turtle")

    entities_df = pd.read_csv(entities_csv)
    alignment_graph, mapping_df = link_entities(graph, entities_df)

    # Save alignment
    align_path = Path(alignment_output)
    align_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_graph.serialize(destination=str(align_path), format="turtle")
    logger.info(f"Alignment saved to {align_path} ({len(alignment_graph)} triples)")

    # Save mapping table
    map_path = Path(mapping_output)
    mapping_df.to_csv(map_path, index=False)
    logger.info(f"Mapping table saved to {map_path}")

    return align_path, map_path


if __name__ == "__main__":
    run_entity_linking()
