"""
RAG engine: NL -> SPARQL generation over RDF/SPARQL using a local LLM (Ollama).
Includes schema summarization, SPARQL generation, execution, and self-repair loop.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"
MAX_PREDICATES = 80
MAX_CLASSES = 40
SAMPLE_TRIPLES = 20
MAX_REPAIR_ATTEMPTS = 3


# ----------------------------
# 0) Utility: call local LLM (Ollama)
# ----------------------------
def ask_local_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to a local Ollama model using the REST API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
    data = response.json()
    return data.get("response", "")


# ----------------------------
# 1) Load RDF graph
# ----------------------------
def load_graph(path: str) -> Graph:
    """Load an RDF graph, auto-detecting format from extension."""
    g = Graph()
    p = Path(path)
    fmt = "nt" if p.suffix == ".nt" else "turtle"
    g.parse(str(p), format=fmt)
    logger.info(f"Loaded {len(g)} triples from {p}")
    return g


# ----------------------------
# 2) Build a small schema summary
# ----------------------------
def get_prefix_block(g: Graph) -> str:
    """Collect prefixes registered in the graph's namespace manager."""
    defaults = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "aib": "http://example.org/ai-news/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in ns_map.items()]
    return "\n".join(sorted(lines))


def list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?p WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?cls WHERE {{
        ?s a ?cls .
    }} LIMIT {limit}
    """
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    SELECT ?s ?p ?o WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    """Assemble a schema summary for LLM prompting."""
    prefixes = get_prefix_block(g)
    preds = list_distinct_predicates(g)
    clss = list_distinct_classes(g)
    samples = sample_triples(g)

    pred_lines = "\n".join(f"- {p}" for p in preds)
    cls_lines = "\n".join(f"- {c}" for c in clss)
    sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)

    summary = f"""{prefixes}

# Predicates (sampled, unique up to {MAX_PREDICATES})
{pred_lines}

# Classes / rdf:type (sampled, unique up to {MAX_CLASSES})
{cls_lines}

# Sample triples (up to {SAMPLE_TRIPLES})
{sample_lines}
"""
    return summary.strip()


# ----------------------------
# 3) Prompting: NL -> SPARQL
# ----------------------------
SPARQL_INSTRUCTIONS = """
You are a SPARQL generator. Convert the user QUESTION into a valid SPARQL 1.1 SELECT query
for the given RDF graph schema. Follow strictly:

- Use ONLY the IRIs/prefixes visible in the SCHEMA SUMMARY.
- Prefer readable SELECT projections with variable names.
- Do NOT invent new predicates/classes.
- Return ONLY the SPARQL query in a single fenced code block labeled ```sparql
- No explanations or extra text outside the code block.
"""

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION:
{question}

Return only the SPARQL query in a code block.
"""


def extract_sparql_from_text(text: str) -> str:
    """Extract the first code block content; fallback to whole text."""
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def generate_sparql(question: str, schema_summary: str, model: str = DEFAULT_MODEL) -> str:
    """Generate a SPARQL query from a natural language question."""
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question), model=model)
    query = extract_sparql_from_text(raw)
    return query


# ----------------------------
# 4) Execute SPARQL with rdflib (and self-repair)
# ----------------------------
def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    """Execute a SPARQL query on the rdflib graph."""
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """
The previous SPARQL failed to execute. Using the SCHEMA SUMMARY and the ERROR MESSAGE,
return a corrected SPARQL 1.1 SELECT query. Follow strictly:

- Use only known prefixes/IRIs.
- Keep it as simple and robust as possible.
- Return only a single code block with the corrected SPARQL.
"""


def repair_sparql(
    schema_summary: str,
    question: str,
    bad_query: str,
    error_msg: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to fix a broken SPARQL query."""
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

ORIGINAL QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR MESSAGE:
{error_msg}

Return only the corrected SPARQL in a code block.
"""
    raw = ask_local_llm(prompt, model=model)
    return extract_sparql_from_text(raw)


def answer_with_sparql_generation(
    g: Graph,
    schema_summary: str,
    question: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_REPAIR_ATTEMPTS,
) -> dict:
    """
    Full RAG pipeline: generate SPARQL -> execute -> self-repair loop.
    Returns dict with keys: query, vars, rows, repaired, attempts, error.
    """
    sparql = generate_sparql(question, schema_summary, model=model)
    attempts = 1

    # Try executing the generated query
    try:
        vars_, rows = run_sparql(g, sparql)
        return {
            "query": sparql,
            "vars": vars_,
            "rows": rows,
            "repaired": False,
            "attempts": attempts,
            "error": None,
        }
    except Exception as e:
        last_error = str(e)
        last_query = sparql

    # Self-repair loop
    for i in range(max_retries):
        attempts += 1
        logger.info(f"Self-repair attempt {i + 1}/{max_retries}")
        try:
            repaired = repair_sparql(schema_summary, question, last_query, last_error, model=model)
            vars_, rows = run_sparql(g, repaired)
            return {
                "query": repaired,
                "vars": vars_,
                "rows": rows,
                "repaired": True,
                "attempts": attempts,
                "error": None,
            }
        except Exception as e2:
            last_error = str(e2)
            last_query = repaired if "repaired" in dir() else last_query

    return {
        "query": last_query,
        "vars": [],
        "rows": [],
        "repaired": True,
        "attempts": attempts,
        "error": last_error,
    }


# ----------------------------
# 5) Baseline: Direct LLM answer without KG
# ----------------------------
def answer_no_rag(question: str, model: str = DEFAULT_MODEL) -> str:
    """Ask the LLM directly without any KB context (baseline)."""
    prompt = f"Answer the following question as best as you can:\n\n{question}"
    return ask_local_llm(prompt, model=model)
