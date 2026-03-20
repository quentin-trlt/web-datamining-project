"""
Named Entity Recognition and Relation Extraction using spaCy.
Reads crawler_output.jsonl and produces extracted_knowledge.csv.
"""

import csv
import json
import logging
from pathlib import Path

import pandas as pd
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Entity types relevant to the AI Bubble domain
RELEVANT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT", "NORP"}


def load_spacy_model(model_name: str = "en_core_web_trf") -> spacy.Language:
    """Load spaCy model, falling back to en_core_web_sm if transformer model unavailable."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(f"{model_name} not found, falling back to en_core_web_sm")
        return spacy.load("en_core_web_sm")


def load_crawled_data(input_path: str = "data/crawler_output.jsonl") -> list[dict]:
    """Load crawled pages from JSONL file."""
    pages = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    logger.info(f"Loaded {len(pages)} pages from {input_path}")
    return pages


def extract_entities(nlp: spacy.Language, text: str) -> list[dict]:
    """Extract named entities from text using spaCy NER."""
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue

        # Normalize: strip whitespace, skip very short entities
        name = ent.text.strip()
        if len(name) < 2:
            continue

        # Deduplicate within the same text
        key = (name.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)

        entities.append({
            "entity": name,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })

    return entities


def extract_relations(nlp: spacy.Language, text: str) -> list[dict]:
    """
    Extract candidate relations between entities co-occurring in the same sentence.
    Uses dependency parsing to find verb connections between entity pairs.
    """
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        # Collect entities in this sentence
        sent_ents = [ent for ent in sent.ents if ent.label_ in RELEVANT_ENTITY_TYPES]
        if len(sent_ents) < 2:
            continue

        # Find the root verb of the sentence
        root_verb = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root_verb = token.lemma_
                break

        # Create relations between entity pairs
        for i, subj in enumerate(sent_ents):
            for obj in sent_ents[i + 1:]:
                # Try to find a connecting verb via dependency path
                relation = _find_connecting_verb(subj, obj, sent)
                if relation is None:
                    relation = root_verb if root_verb else "related_to"

                relations.append({
                    "subject": subj.text.strip(),
                    "subject_label": subj.label_,
                    "predicate": relation,
                    "object": obj.text.strip(),
                    "object_label": obj.label_,
                    "sentence": sent.text.strip()[:200],
                })

    return relations


def _find_connecting_verb(ent1, ent2, sent) -> str | None:
    """Try to find a verb that connects two entities via dependency arcs."""
    ent1_tokens = {token.i for token in ent1}
    ent2_tokens = {token.i for token in ent2}

    for token in sent:
        if token.pos_ != "VERB":
            continue
        # Check if this verb has a subject from ent1 and object from ent2 (or vice versa)
        children = list(token.children)
        has_subj = any(
            child.i in ent1_tokens or child.head.i in ent1_tokens
            for child in children
            if child.dep_ in ("nsubj", "nsubjpass")
        )
        has_obj = any(
            child.i in ent2_tokens or child.head.i in ent2_tokens
            for child in children
            if child.dep_ in ("dobj", "pobj", "attr")
        )
        if has_subj and has_obj:
            return token.lemma_

    return None


def run_extraction(
    input_path: str = "data/crawler_output.jsonl",
    entities_output: str = "data/extracted_entities.csv",
    relations_output: str = "data/extracted_relations.csv",
) -> tuple[Path, Path]:
    """
    Run the full NER + relation extraction pipeline.

    Returns paths to the entities and relations CSV files.
    """
    nlp = load_spacy_model()
    pages = load_crawled_data(input_path)

    all_entities = []
    all_relations = []

    for page in pages:
        url = page["url"]
        text = page["text"]
        logger.info(f"Extracting from: {url}")

        # NER
        entities = extract_entities(nlp, text)
        for ent in entities:
            ent["source_url"] = url
        all_entities.extend(entities)

        # Relations
        relations = extract_relations(nlp, text)
        for rel in relations:
            rel["source_url"] = url
        all_relations.extend(relations)

        logger.info(f"  Found {len(entities)} entities, {len(relations)} relations")

    # Save entities CSV
    ent_path = Path(entities_output)
    ent_path.parent.mkdir(parents=True, exist_ok=True)
    df_entities = pd.DataFrame(all_entities)
    df_entities.to_csv(ent_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Entities saved to {ent_path} ({len(all_entities)} total)")

    # Save relations CSV
    rel_path = Path(relations_output)
    df_relations = pd.DataFrame(all_relations)
    df_relations.to_csv(rel_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Relations saved to {rel_path} ({len(all_relations)} total)")

    return ent_path, rel_path


if __name__ == "__main__":
    run_extraction()
