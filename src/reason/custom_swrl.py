"""
Custom SWRL reasoning on the AI-News KB:
Rule: Organization(?o) ∧ develops(?o, ?p) ∧ AIProduct(?p) → AICompany(?o)
"""

import csv
import logging
from pathlib import Path

import owlready2
from rdflib import Graph, Namespace, RDF, RDFS, URIRef

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIB_NS = "http://example.org/ai-news/"


def build_ai_ontology_owl(expanded_nt: str = "kg_artifacts/expanded.nt") -> owlready2.Ontology:
    """
    Build an OWL ontology from the expanded KB for SWRL reasoning.
    Extracts organizations, products, and develops relationships.
    """
    # Load expanded KB with rdflib to extract relevant triples
    rdf_graph = Graph()
    nt_path = Path(expanded_nt)
    if nt_path.exists() and nt_path.stat().st_size > 0:
        rdf_graph.parse(str(nt_path), format="nt")
    else:
        logger.warning(f"Expanded KB not found or empty at {expanded_nt}, using initial KB")
        initial_path = Path("kg_artifacts/initial_kb.ttl")
        if initial_path.exists():
            rdf_graph.parse(str(initial_path), format="turtle")

    logger.info(f"Loaded {len(rdf_graph)} triples from KB")

    # Create OWL ontology with owlready2
    onto = owlready2.get_ontology("http://example.org/ai-news-reasoning#")

    with onto:
        # Define classes
        class Organization(owlready2.Thing):
            pass

        class AICompany(Organization):
            pass

        class Product(owlready2.Thing):
            pass

        class AIProduct(Product):
            pass

        class Person(owlready2.Thing):
            pass

        class Technology(owlready2.Thing):
            pass

        # Define object properties
        class develops(owlready2.ObjectProperty):
            domain = [Organization]
            range = [Product]

        class worksFor(owlready2.ObjectProperty):
            domain = [Person]
            range = [Organization]

        class uses(owlready2.ObjectProperty):
            domain = [Organization]
            range = [Technology]

    # Populate with individuals from the KB
    AIB = Namespace(AIB_NS)
    org_type = AIB.Organization
    product_type = AIB.Product
    ai_product_type = AIB.AIProduct
    develops_pred = AIB.develops

    with onto:
        # Find organizations
        orgs_found = set()
        for s in rdf_graph.subjects(RDF.type, org_type):
            name = str(s).split("/")[-1]
            if name and name not in orgs_found:
                orgs_found.add(name)
                ind = Organization(name)

        # Find products
        products_found = set()
        for s in rdf_graph.subjects(RDF.type, product_type):
            name = str(s).split("/")[-1]
            if name and name not in products_found:
                products_found.add(name)
                ind = Product(name)

        # Mark AI-related products as AIProduct
        ai_keywords = ["ai", "gpt", "chatgpt", "copilot", "gemini", "claude", "llm",
                        "bert", "dall-e", "midjourney", "siri", "alexa", "cortana",
                        "watson", "bard", "machine_learning", "deep_learning", "neural"]
        for prod_name in products_found:
            if any(kw in prod_name.lower() for kw in ai_keywords):
                ind = onto[prod_name]
                if ind:
                    ind.is_a.append(AIProduct)

        # Add develops relationships
        for s, p, o in rdf_graph.triples((None, develops_pred, None)):
            subj_name = str(s).split("/")[-1]
            obj_name = str(o).split("/")[-1]
            subj_ind = onto[subj_name]
            obj_ind = onto[obj_name]
            if subj_ind and obj_ind:
                subj_ind.develops.append(obj_ind)

        # If no develops relations found, add some known ones
        if not list(rdf_graph.triples((None, develops_pred, None))):
            logger.info("No develops relations in KB, adding known AI company-product pairs")
            known_pairs = [
                ("OpenAI", "ChatGPT", True),
                ("Google", "Gemini", True),
                ("Microsoft", "Copilot", True),
                ("Meta", "LLaMA", True),
                ("Apple", "Siri", True),
                ("Amazon", "Alexa", True),
                ("IBM", "Watson", True),
                ("Tesla", "Autopilot", True),
            ]
            for org_name, prod_name, is_ai in known_pairs:
                org_ind = Organization(org_name)
                prod_ind = AIProduct(prod_name) if is_ai else Product(prod_name)
                org_ind.develops.append(prod_ind)

    logger.info(f"Ontology built: {len(list(onto.individuals()))} individuals")
    logger.info(f"  Organizations: {len(orgs_found) if orgs_found else 'using known pairs'}")
    logger.info(f"  Products: {len(products_found) if products_found else 'using known pairs'}")

    return onto


def add_ai_company_rule(onto: owlready2.Ontology) -> None:
    """
    SWRL rule: Organization(?o) ∧ develops(?o, ?p) ∧ AIProduct(?p) → AICompany(?o)
    """
    with onto:
        rule = owlready2.Imp()
        rule.set_as_rule(
            "Organization(?o), develops(?o, ?p), AIProduct(?p) -> AICompany(?o)"
        )
    logger.info("SWRL rule added: Organization(?o) ∧ develops(?o, ?p) ∧ AIProduct(?p) → AICompany(?o)")


def run_reasoner_and_display(onto: owlready2.Ontology) -> list[str]:
    """Run the reasoner and display inferred AICompany instances."""
    logger.info("Running reasoner (Pellet)...")
    try:
        owlready2.sync_reasoner_pellet(
            infer_property_values=True,
            infer_data_property_values=True,
        )
    except Exception as e:
        logger.warning(f"Pellet failed, trying HermiT: {e}")
        owlready2.sync_reasoner()

    # Query AICompany instances
    ai_company_class = onto["AICompany"]
    if ai_company_class is None:
        logger.error("AICompany class not found")
        return []

    inferred = list(ai_company_class.instances())

    logger.info(f"\n{'=' * 50}")
    logger.info(f"SWRL Reasoning Results — AICompany")
    logger.info(f"{'=' * 50}")

    results = []
    for ind in inferred:
        products = ind.develops if hasattr(ind, "develops") else []
        ai_products = [p.name for p in products if onto["AIProduct"] in p.is_a]
        logger.info(f"  {ind.name} → AICompany ✓ (develops: {', '.join(ai_products) if ai_products else 'N/A'})")
        results.append(ind.name)

    logger.info(f"\nTotal inferred AICompany instances: {len(results)}")
    return results


def run_custom_swrl(
    expanded_nt: str = "kg_artifacts/expanded.nt",
    output_path: str = "data/swrl_inferred_results.csv",
) -> list[str]:
    """Run the complete custom SWRL demonstration."""
    logger.info("=" * 60)
    logger.info("Part 2: Custom SWRL Reasoning on AI-News KB")
    logger.info("=" * 60)

    onto = build_ai_ontology_owl(expanded_nt)
    add_ai_company_rule(onto)
    results = run_reasoner_and_display(onto)

    # Save results
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "inferred_class", "rule"])
        for name in results:
            writer.writerow([name, "AICompany", "Organization(?o) ∧ develops(?o, ?p) ∧ AIProduct(?p) → AICompany(?o)"])
    logger.info(f"Results saved to {out}")

    return results


if __name__ == "__main__":
    run_custom_swrl()
