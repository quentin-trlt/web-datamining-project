"""
KB Expansion via Wikidata SPARQL: 1-hop and 2-hop expansion from aligned entities.
Target: 50k–200k triples, 5k–30k entities, 50–200 relations.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB = Namespace("http://example.org/ai-news/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Allowlist of meaningful Wikidata properties for expansion
ALLOWED_PROPERTIES = {
    "P31",   # instance of
    "P279",  # subclass of
    "P17",   # country
    "P159",  # headquarters location
    "P112",  # founded by
    "P127",  # owned by
    "P1056", # product or material produced
    "P452",  # industry
    "P169",  # chief executive officer
    "P108",  # employer
    "P166",  # award received
    "P27",   # country of citizenship
    "P19",   # place of birth
    "P69",   # educated at
    "P39",   # position held
    "P463",  # member of
    "P355",  # subsidiary
    "P749",  # parent organization
    "P1454", # legal form
    "P571",  # inception
    "P576",  # dissolved
    "P137",  # operator
    "P178",  # developer
    "P176",  # manufacturer
    "P361",  # part of
    "P527",  # has part
    "P131",  # located in admin entity
    "P36",   # capital
    "P150",  # contains admin entity
    "P47",   # shares border with
    "P530",  # diplomatic relation
    "P37",   # official language
    "P38",   # currency
    "P6",    # head of government
    "P35",   # head of state
    "P2283", # uses
    "P101",  # field of work
    "P106",  # occupation
    "P800",  # notable work
    "P1830", # owner of
    "P1376", # capital of
    "P910",  # topic's main category
}


def _sparql_query(query: str) -> list[dict]:
    """Execute a SPARQL query against the Wikidata endpoint."""
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    sparql.addCustomHttpHeader("User-Agent", "AIBubbleResearchBot/1.0 (academic project)")

    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.warning(f"SPARQL query failed: {e}")
        return []


def _sparql_1hop(entity_qid: str, limit: int = 200) -> list[tuple[str, str]]:
    """Get 1-hop neighbors: all (predicate, object) for a Wikidata entity."""
    query = f"""
    SELECT ?p ?o WHERE {{
        wd:{entity_qid} ?p ?o .
        FILTER(isIRI(?o))
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    }}
    LIMIT {limit}
    """
    results = _sparql_query(query)
    triples = []
    for r in results:
        pred_uri = r["p"]["value"]
        obj_uri = r["o"]["value"]
        # Extract property ID (e.g., P31 from full URI)
        prop_id = pred_uri.split("/")[-1]
        if prop_id in ALLOWED_PROPERTIES:
            triples.append((pred_uri, obj_uri))
    return triples


def _sparql_predicate_expansion(prop_id: str, limit: int = 5000) -> list[tuple[str, str, str]]:
    """Get all triples for a specific predicate (predicate-controlled expansion)."""
    query = f"""
    SELECT ?s ?o WHERE {{
        ?s wdt:{prop_id} ?o .
        FILTER(isIRI(?o))
    }}
    LIMIT {limit}
    """
    results = _sparql_query(query)
    triples = []
    pred_uri = str(WDT[prop_id])
    for r in results:
        subj_uri = r["s"]["value"]
        obj_uri = r["o"]["value"]
        triples.append((subj_uri, pred_uri, obj_uri))
    return triples


def _sparql_2hop(entity_qid: str, limit: int = 500) -> list[tuple[str, str, str]]:
    """Get 2-hop expansion: entity -> mid -> object."""
    query = f"""
    SELECT ?p1 ?mid ?p2 ?o WHERE {{
        wd:{entity_qid} ?p1 ?mid .
        ?mid ?p2 ?o .
        FILTER(isIRI(?mid) && isIRI(?o))
        FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
        FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
    }}
    LIMIT {limit}
    """
    results = _sparql_query(query)
    triples = []
    for r in results:
        p1_id = r["p1"]["value"].split("/")[-1]
        p2_id = r["p2"]["value"].split("/")[-1]
        if p1_id in ALLOWED_PROPERTIES and p2_id in ALLOWED_PROPERTIES:
            mid_uri = r["mid"]["value"]
            obj_uri = r["o"]["value"]
            # Add both hops as triples
            triples.append((f"http://www.wikidata.org/entity/{entity_qid}", r["p1"]["value"], mid_uri))
            triples.append((mid_uri, r["p2"]["value"], obj_uri))
    return triples


def _load_cache(cache_path: Path) -> dict:
    """Load expansion cache from disk."""
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Save expansion cache to disk."""
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def expand_kb(
    alignment_graph: Graph,
    mapping_df: pd.DataFrame,
    max_triples: int = 150000,
    cache_path: str = "data/wikidata_cache.json",
) -> Graph:
    """
    Expand the KB by querying Wikidata for aligned entities.
    Uses 1-hop, 2-hop, and predicate-controlled expansion.
    """
    expansion_graph = Graph()
    expansion_graph.bind("wd", WD)
    expansion_graph.bind("wdt", WDT)

    cache = _load_cache(Path(cache_path))
    triple_set = set()

    # Get matched entities with their Wikidata IDs
    matched = mapping_df[mapping_df["matched"] == True]
    entity_qids = list(matched["wikidata_id"].unique())
    logger.info(f"Starting expansion from {len(entity_qids)} aligned entities")

    # Phase 1: 1-hop expansion for all aligned entities
    logger.info("Phase 1: 1-hop expansion")
    for idx, qid in enumerate(entity_qids):
        if len(triple_set) >= max_triples:
            break

        cache_key = f"1hop_{qid}"
        if cache_key in cache:
            hop_results = cache[cache_key]
        else:
            hop_results = [(p, o) for p, o in _sparql_1hop(qid)]
            cache[cache_key] = hop_results
            time.sleep(1)

        for pred_uri, obj_uri in hop_results:
            triple = (f"http://www.wikidata.org/entity/{qid}", pred_uri, obj_uri)
            if triple not in triple_set:
                triple_set.add(triple)
                expansion_graph.add((URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])))

        if (idx + 1) % 10 == 0:
            _save_cache(cache, Path(cache_path))
            logger.info(f"  1-hop: {idx + 1}/{len(entity_qids)} entities, {len(triple_set)} triples")

    logger.info(f"After 1-hop: {len(triple_set)} triples")

    # Phase 2: Predicate-controlled expansion for key predicates
    if len(triple_set) < max_triples:
        logger.info("Phase 2: Predicate-controlled expansion")
        key_predicates = ["P31", "P279", "P17", "P159", "P452", "P178", "P112", "P108", "P166"]
        for prop_id in key_predicates:
            if len(triple_set) >= max_triples:
                break

            cache_key = f"pred_{prop_id}"
            if cache_key in cache:
                pred_triples = [tuple(t) for t in cache[cache_key]]
            else:
                limit = min(10000, max_triples - len(triple_set))
                pred_triples = _sparql_predicate_expansion(prop_id, limit=limit)
                cache[cache_key] = pred_triples
                time.sleep(2)

            added = 0
            for s, p, o in pred_triples:
                triple = (s, p, o)
                if triple not in triple_set:
                    triple_set.add(triple)
                    expansion_graph.add((URIRef(s), URIRef(p), URIRef(o)))
                    added += 1

            logger.info(f"  Predicate wdt:{prop_id}: +{added} triples (total: {len(triple_set)})")

    # Phase 3: 2-hop expansion if still below target
    if len(triple_set) < max_triples:
        logger.info("Phase 3: 2-hop expansion")
        for idx, qid in enumerate(entity_qids[:20]):  # limit to top 20 entities
            if len(triple_set) >= max_triples:
                break

            cache_key = f"2hop_{qid}"
            if cache_key in cache:
                hop2_triples = [tuple(t) for t in cache[cache_key]]
            else:
                hop2_triples = _sparql_2hop(qid)
                cache[cache_key] = hop2_triples
                time.sleep(2)

            added = 0
            for s, p, o in hop2_triples:
                triple = (s, p, o)
                if triple not in triple_set:
                    triple_set.add(triple)
                    expansion_graph.add((URIRef(s), URIRef(p), URIRef(o)))
                    added += 1

            if added > 0:
                logger.info(f"  2-hop {qid}: +{added} triples (total: {len(triple_set)})")

    _save_cache(cache, Path(cache_path))
    logger.info(f"Expansion complete: {len(triple_set)} total triples")
    return expansion_graph


def clean_expanded_graph(graph: Graph) -> Graph:
    """Clean the expanded graph: remove duplicates and ensure consistency."""
    cleaned = Graph()
    for ns_prefix, ns_uri in graph.namespaces():
        cleaned.bind(ns_prefix, ns_uri)

    seen = set()
    for s, p, o in graph:
        triple = (str(s), str(p), str(o))
        if triple not in seen and isinstance(o, URIRef):
            seen.add(triple)
            cleaned.add((s, p, o))

    removed = len(graph) - len(cleaned)
    if removed > 0:
        logger.info(f"Cleaning removed {removed} triples (duplicates/literals)")
    return cleaned


def compute_statistics(graph: Graph) -> dict:
    """Compute KB statistics."""
    subjects = set()
    predicates = set()
    objects = set()

    for s, p, o in graph:
        subjects.add(str(s))
        predicates.add(str(p))
        if isinstance(o, URIRef):
            objects.add(str(o))

    entities = subjects | objects
    stats = {
        "total_triples": len(graph),
        "unique_entities": len(entities),
        "unique_relations": len(predicates),
        "unique_subjects": len(subjects),
        "unique_objects": len(objects),
    }

    logger.info(f"KB Statistics:")
    for key, val in stats.items():
        logger.info(f"  {key}: {val}")

    return stats


def run_expansion(
    alignment_ttl: str = "kg_artifacts/alignment.ttl",
    initial_kb_path: str = "kg_artifacts/initial_kb.ttl",
    mapping_csv: str = "data/entity_mapping.csv",
    output_nt: str = "kg_artifacts/expanded.nt",
    stats_output: str = "data/kb_statistics.json",
    max_triples: int = 150000,
) -> Path:
    """Run the full KB expansion pipeline."""
    # Load alignment graph and mapping
    alignment_graph = Graph()
    if Path(alignment_ttl).exists():
        alignment_graph.parse(alignment_ttl, format="turtle")

    mapping_df = pd.read_csv(mapping_csv)

    # Expand
    expansion_graph = expand_kb(alignment_graph, mapping_df, max_triples=max_triples)

    # Merge with initial KB
    initial_graph = Graph()
    if Path(initial_kb_path).exists():
        initial_graph.parse(initial_kb_path, format="turtle")

    merged = Graph()
    for ns_prefix, ns_uri in expansion_graph.namespaces():
        merged.bind(ns_prefix, ns_uri)

    for triple in initial_graph:
        merged.add(triple)
    for triple in alignment_graph:
        merged.add(triple)
    for triple in expansion_graph:
        merged.add(triple)

    # Clean
    merged = clean_expanded_graph(merged)

    # Statistics
    stats = compute_statistics(merged)

    # Save expanded KB
    out = Path(output_nt)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.serialize(destination=str(out), format="nt")
    logger.info(f"Expanded KB saved to {out}")

    # Save statistics
    stats_path = Path(stats_output)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")

    return out


if __name__ == "__main__":
    run_expansion()
