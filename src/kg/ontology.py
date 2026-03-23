"""
Define the AI-News ontology: classes, object properties, and datatype properties.
Serializes to kg_artifacts/ontology.ttl.
"""

import logging
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD
from rdflib.namespace import DCTERMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB = Namespace("http://example.org/ai-news/")
SCHEMA = Namespace("http://schema.org/")


def build_ontology() -> Graph:
    """Build the AI-News domain ontology with classes and properties."""
    g = Graph()
    g.bind("aib", AIB)
    g.bind("schema", SCHEMA)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dcterms", DCTERMS)

    # ── Classes ──────────────────────────────────────────────────────────
    classes = {
        "Person": ("A person mentioned in AI news", None),
        "Organization": ("An organization or company", None),
        "AICompany": ("An organization that develops AI products", "Organization"),
        "TechCompany": ("A technology company", "Organization"),
        "Product": ("A product or service", None),
        "AIProduct": ("An AI-powered product or service", "Product"),
        "Technology": ("A technology or technique", None),
        "Event": ("A notable event", None),
        "Location": ("A geographic location", None),
        "Country": ("A country", "Location"),
        "NationalGroup": ("A national, religious, or political group", None),
    }
    for name, (comment, parent) in classes.items():
        cls_uri = AIB[name]
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(name)))
        g.add((cls_uri, RDFS.comment, Literal(comment)))
        if parent:
            g.add((cls_uri, RDFS.subClassOf, AIB[parent]))

    # ── Object Properties ────────────────────────────────────────────────
    object_props = {
        "developedBy": ("Product", "Organization", "Developed by an organization"),
        "headquarteredIn": ("Organization", "Location", "Headquarters location"),
        "foundedBy": ("Organization", "Person", "Founded by a person"),
        "worksFor": ("Person", "Organization", "Person works for organization"),
        "locatedIn": ("Event", "Location", "Event location"),
        "investedIn": ("Organization", "Organization", "Investment relationship"),
        "uses": ("Organization", "Technology", "Organization uses a technology"),
        "develops": ("Organization", "Product", "Organization develops a product"),
        "relatedTo": (None, None, "Generic relation between entities"),
    }
    for name, (domain, range_, comment) in object_props.items():
        prop_uri = AIB[name]
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(name)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        if domain:
            g.add((prop_uri, RDFS.domain, AIB[domain]))
        if range_:
            g.add((prop_uri, RDFS.range, AIB[range_]))

    # ── Datatype Properties ──────────────────────────────────────────────
    data_props = {
        "foundedYear": ("Organization", XSD.gYear, "Year the organization was founded"),
        "revenue": ("Organization", XSD.decimal, "Revenue amount"),
        "employeeCount": ("Organization", XSD.integer, "Number of employees"),
    }
    for name, (domain, range_, comment) in data_props.items():
        prop_uri = AIB[name]
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.label, Literal(name)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        g.add((prop_uri, RDFS.domain, AIB[domain]))
        g.add((prop_uri, RDFS.range, range_))

    logger.info(f"Ontology built: {len(g)} triples")
    return g


def run_ontology(output_path: str = "kg_artifacts/ontology.ttl") -> Path:
    """Build ontology and serialize to Turtle."""
    g = build_ontology()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    logger.info(f"Ontology saved to {out}")
    return out


if __name__ == "__main__":
    run_ontology()
