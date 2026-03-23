"""
Build the initial RDF Knowledge Base from extracted entities and relations CSVs.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB = Namespace("http://example.org/ai-news/")
SCHEMA = Namespace("http://schema.org/")

# Map spaCy entity labels to RDF classes
LABEL_TO_CLASS = {
    "PERSON": AIB.Person,
    "ORG": AIB.Organization,
    "GPE": AIB.Location,
    "PRODUCT": AIB.Product,
    "EVENT": AIB.Event,
    "DATE": AIB.Event,  # dates treated as temporal entities
    "MONEY": SCHEMA.MonetaryAmount,
    "NORP": AIB.NationalGroup,
}


def _normalize_uri(name: str) -> str:
    """Convert an entity name to a URI-safe slug."""
    slug = name.strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug


def _normalize_predicate(verb: str) -> str:
    """Convert a predicate string to camelCase."""
    words = re.sub(r"[^\w\s]", "", verb.strip()).split()
    if not words:
        return "relatedTo"
    result = words[0].lower()
    for w in words[1:]:
        result += w.capitalize()
    return result


def load_entities(entities_csv: Path) -> pd.DataFrame:
    """Load and deduplicate the extracted entities CSV."""
    df = pd.read_csv(entities_csv)
    df = df.drop_duplicates(subset=["entity", "label"])
    logger.info(f"Loaded {len(df)} unique entities from {entities_csv}")
    return df


def load_relations(relations_csv: Path) -> pd.DataFrame:
    """Load the extracted relations CSV."""
    df = pd.read_csv(relations_csv)
    logger.info(f"Loaded {len(df)} relations from {relations_csv}")
    return df


def build_initial_graph(entities_df: pd.DataFrame, relations_df: pd.DataFrame) -> Graph:
    """Build an RDF graph from extracted entities and relations."""
    g = Graph()
    g.bind("aib", AIB)
    g.bind("schema", SCHEMA)
    g.bind("rdfs", RDFS)
    g.bind("dcterms", DCTERMS)

    # Add entities with type and label
    entity_uris = {}
    for _, row in entities_df.iterrows():
        name = str(row["entity"])
        label = str(row["label"])
        slug = _normalize_uri(name)
        if not slug:
            continue

        uri = AIB[slug]
        entity_uris[(name.lower(), label)] = uri

        rdf_class = LABEL_TO_CLASS.get(label, AIB.Thing)
        g.add((uri, RDF.type, rdf_class))
        g.add((uri, RDFS.label, Literal(name)))

        # Provenance
        if "source_url" in row and pd.notna(row["source_url"]):
            g.add((uri, DCTERMS.source, URIRef(row["source_url"])))

    # Add relations as triples
    for _, row in relations_df.iterrows():
        subj_name = str(row["subject"])
        obj_name = str(row["object"])
        predicate_str = str(row["predicate"])

        subj_slug = _normalize_uri(subj_name)
        obj_slug = _normalize_uri(obj_name)
        pred_slug = _normalize_predicate(predicate_str)

        if not subj_slug or not obj_slug or not pred_slug:
            continue

        subj_uri = AIB[subj_slug]
        pred_uri = AIB[pred_slug]
        obj_uri = AIB[obj_slug]

        g.add((subj_uri, pred_uri, obj_uri))

        # Ensure subject and object have at least a label
        if (subj_uri, RDFS.label, None) not in g:
            g.add((subj_uri, RDFS.label, Literal(subj_name)))
        if (obj_uri, RDFS.label, None) not in g:
            g.add((obj_uri, RDFS.label, Literal(obj_name)))

    logger.info(f"Initial KB built: {len(g)} triples")

    # Count unique entities
    subjects = set(g.subjects())
    objects = set(g.objects())
    entities = {s for s in subjects if isinstance(s, URIRef)} | {
        o for o in objects if isinstance(o, URIRef)
    }
    logger.info(f"Unique entities: {len(entities)}")

    return g


def build_kb(
    entities_csv: str = "data/extracted_entities.csv",
    relations_csv: str = "data/extracted_relations.csv",
    output_path: str = "kg_artifacts/initial_kb.ttl",
) -> Path:
    """Run the full KB build pipeline."""
    ent_df = load_entities(Path(entities_csv))
    rel_df = load_relations(Path(relations_csv))
    g = build_initial_graph(ent_df, rel_df)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    logger.info(f"Initial KB saved to {out}")
    return out


if __name__ == "__main__":
    build_kb()
