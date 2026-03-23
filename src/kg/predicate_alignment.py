"""
Predicate alignment: match private KB predicates to Wikidata properties using SPARQL.
"""

import logging
import time
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, OWL, RDFS, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB = Namespace("http://example.org/ai-news/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Manual predicate mapping hints for common predicates
KNOWN_MAPPINGS = {
    "developedBy": ("P178", "developer"),
    "headquarteredIn": ("P159", "headquarters location"),
    "foundedBy": ("P112", "founded by"),
    "worksFor": ("P108", "employer"),
    "locatedIn": ("P131", "located in the administrative territorial entity"),
    "investedIn": ("P1830", "owner of"),
    "uses": ("P2283", "uses"),
    "develops": ("P1056", "product or material produced or service provided"),
}


def _search_wikidata_property(predicate_name: str) -> list[dict]:
    """Search Wikidata SPARQL endpoint for properties matching the predicate name."""
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.setReturnFormat(JSON)

    # Search by label containing the predicate words
    search_term = predicate_name.lower()
    # Split camelCase into words for search
    import re
    words = re.sub(r"([A-Z])", r" \1", predicate_name).lower().strip()

    query = f"""
    SELECT ?property ?propertyLabel WHERE {{
        ?property a wikibase:Property .
        ?property rdfs:label ?propertyLabel .
        FILTER(CONTAINS(LCASE(?propertyLabel), "{words}"))
        FILTER(LANG(?propertyLabel) = "en")
    }}
    LIMIT 10
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        candidates = []
        for binding in results["results"]["bindings"]:
            prop_uri = binding["property"]["value"]
            prop_label = binding["propertyLabel"]["value"]
            prop_id = prop_uri.split("/")[-1]
            candidates.append({
                "property_id": prop_id,
                "property_uri": prop_uri,
                "label": prop_label,
            })
        return candidates
    except Exception as e:
        logger.warning(f"SPARQL query failed for '{predicate_name}': {e}")
        return []


def align_predicates(
    graph: Graph,
    alignment_graph: Graph,
) -> tuple[Graph, pd.DataFrame]:
    """
    Align private predicates with Wikidata properties.
    Returns updated alignment graph and a mapping DataFrame.
    """
    alignment_graph.bind("wdt", WDT)

    # Collect unique predicates from the main graph in the aib: namespace
    predicates = set()
    for p in graph.predicates():
        if str(p).startswith(str(AIB)):
            local_name = str(p).replace(str(AIB), "")
            predicates.add(local_name)

    mapping_rows = []
    total = len(predicates)
    logger.info(f"Found {total} private predicates to align")

    for idx, pred_name in enumerate(sorted(predicates)):
        logger.info(f"[{idx + 1}/{total}] Aligning predicate: {pred_name}")

        # Check known mappings first
        if pred_name in KNOWN_MAPPINGS:
            prop_id, prop_label = KNOWN_MAPPINGS[pred_name]
            wdt_uri = WDT[prop_id]
            pred_uri = AIB[pred_name]
            alignment_graph.add((pred_uri, OWL.equivalentProperty, wdt_uri))
            mapping_rows.append({
                "predicate": pred_name,
                "wikidata_property": prop_id,
                "wikidata_label": prop_label,
                "alignment_type": "equivalentProperty",
                "source": "known_mapping",
            })
            logger.info(f"  Known mapping: wdt:{prop_id} ({prop_label})")
            continue

        # Query Wikidata SPARQL
        candidates = _search_wikidata_property(pred_name)
        time.sleep(1)  # rate limiting

        if candidates:
            best = candidates[0]
            pred_uri = AIB[pred_name]
            wdt_uri = WDT[best["property_id"]]
            # Use subPropertyOf for SPARQL-discovered matches (less certain)
            alignment_graph.add((pred_uri, RDFS.subPropertyOf, wdt_uri))
            mapping_rows.append({
                "predicate": pred_name,
                "wikidata_property": best["property_id"],
                "wikidata_label": best["label"],
                "alignment_type": "subPropertyOf",
                "source": "sparql_search",
            })
            logger.info(f"  SPARQL match: wdt:{best['property_id']} ({best['label']})")
        else:
            mapping_rows.append({
                "predicate": pred_name,
                "wikidata_property": "",
                "wikidata_label": "",
                "alignment_type": "none",
                "source": "no_match",
            })
            logger.info(f"  No match found")

    mapping_df = pd.DataFrame(mapping_rows)
    aligned = mapping_df[mapping_df["alignment_type"] != "none"]
    logger.info(f"Predicate alignment: {len(aligned)}/{total} aligned")
    return alignment_graph, mapping_df


def run_predicate_alignment(
    initial_kb_path: str = "kg_artifacts/initial_kb.ttl",
    alignment_path: str = "kg_artifacts/alignment.ttl",
    mapping_output: str = "data/predicate_mapping.csv",
) -> Path:
    """Run the full predicate alignment pipeline."""
    graph = Graph()
    graph.parse(initial_kb_path, format="turtle")

    alignment_graph = Graph()
    if Path(alignment_path).exists():
        alignment_graph.parse(alignment_path, format="turtle")

    alignment_graph, mapping_df = align_predicates(graph, alignment_graph)

    # Save updated alignment
    out = Path(alignment_path)
    alignment_graph.serialize(destination=str(out), format="turtle")
    logger.info(f"Updated alignment saved to {out}")

    # Save predicate mapping
    map_path = Path(mapping_output)
    mapping_df.to_csv(map_path, index=False)
    logger.info(f"Predicate mapping saved to {map_path}")

    return out


if __name__ == "__main__":
    run_predicate_alignment()
