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
MAX_PREDICATES = 30
MAX_CLASSES = 15
SAMPLE_TRIPLES = 10
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

    # Fallback to initial_kb.ttl if expanded is empty
    if len(g) == 0:
        from utils import project_path
        fallback = Path(project_path("kg_artifacts/initial_kb.ttl"))
        if fallback.exists():
            logger.warning(f"Graph empty, falling back to {fallback}")
            g.parse(str(fallback), format="turtle")

    logger.info(f"Loaded {len(g)} triples from {p}")
    return g


# ----------------------------
# 2) Build a small schema summary
# ----------------------------
# Only keep prefixes that are actually useful for querying
USEFUL_PREFIXES = {
    "aib": "http://example.org/ai-news/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
}


def get_prefix_block(g: Graph) -> str:
    """Return only useful prefixes for SPARQL queries."""
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(USEFUL_PREFIXES.items())]
    return "\n".join(lines)


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


def _shorten_uri(uri: str, ns_map: dict) -> str:
    """Shorten a full URI to prefixed form using ns_map."""
    for prefix, namespace in sorted(ns_map.items(), key=lambda x: -len(x[1])):
        if uri.startswith(namespace) and prefix:
            local = uri[len(namespace):]
            return f"{prefix}:{local}"
    return f"<{uri}>"


def build_schema_summary(g: Graph) -> str:
    """Assemble a schema summary for LLM prompting, with prefixed URIs."""
    prefixes = get_prefix_block(g)
    # Use USEFUL_PREFIXES for shortening URIs (includes wd: and wdt:)
    ns_map = dict(USEFUL_PREFIXES)

    preds = list_distinct_predicates(g)
    clss = list_distinct_classes(g)
    samples = sample_triples(g)

    pred_lines = "\n".join(f"- {_shorten_uri(p, ns_map)}" for p in preds)
    cls_lines = "\n".join(f"- {_shorten_uri(c, ns_map)}" for c in clss)
    sample_lines = "\n".join(
        f"- {_shorten_uri(s, ns_map)}  {_shorten_uri(p, ns_map)}  {_shorten_uri(o, ns_map)}"
        for s, p, o in samples
    )

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
CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SELECT_RE = re.compile(
    r"((?:PREFIX\s+\S+\s*<[^>]+>\s*)*\s*SELECT\b.*)",
    re.IGNORECASE | re.DOTALL,
)


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    """Build a compact prompt with few-shot examples for small models."""
    return f"""Write a SPARQL SELECT query for a LOCAL RDF graph (not Wikidata).
Available predicates: rdf:type, aib:develops, aib:headquarteredIn, aib:foundedBy, aib:worksFor, aib:uses, aib:locatedIn, aib:investedIn, aib:relatedTo, owl:sameAs, wdt:P112, wdt:P159, wdt:P178, wdt:P452, wdt:P279, wdt:P31, wdt:P17, wdt:P108, wdt:P166, wdt:P527
Available classes: aib:Person, aib:Organization, aib:Product, aib:Location, aib:Event, aib:Technology, aib:NationalGroup

Example 1:
Q: Which organizations develop products?
A:
```sparql
PREFIX aib: <http://example.org/ai-news/>
SELECT DISTINCT ?org ?product WHERE {{ ?org aib:develops ?product . }} LIMIT 20
```

Example 2:
Q: List all people in the knowledge base.
A:
```sparql
PREFIX aib: <http://example.org/ai-news/>
SELECT DISTINCT ?person WHERE {{ ?person a aib:Person . }} LIMIT 20
```

Example 3:
Q: What types of entities exist?
A:
```sparql
SELECT DISTINCT ?type WHERE {{ ?s a ?type . }} LIMIT 20
```

Now answer:
Q: {question}
A:
"""


def extract_sparql_from_text(text: str) -> str:
    """Extract SPARQL from LLM output. Try code block first, then SELECT pattern."""
    # 1) Try fenced code block
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    # 2) Try to find SELECT statement directly
    m = SELECT_RE.search(text)
    if m:
        query = m.group(1).strip()
        # Remove trailing junk after the closing }
        brace_depth = 0
        end_pos = 0
        for i, ch in enumerate(query):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_pos = i + 1
                    break
        if end_pos > 0:
            query = query[:end_pos]
        return query.strip()

    # 3) Last resort: return whole text
    return text.strip()


def sanitize_sparql(query: str) -> str:
    """
    Fix common SPARQL syntax errors produced by small LLMs:
    - ORDER BY / LIMIT / OFFSET inside WHERE {}
    - FILTER with LANG on non-literal variables
    - Missing LIMIT
    """
    # 1) Move ORDER BY / LIMIT / OFFSET / GROUP BY / HAVING outside WHERE {}
    # Find the last closing brace of the WHERE block
    clauses_inside_re = re.compile(
        r"(ORDER\s+BY\b[^\n}]*|LIMIT\s+\d+|OFFSET\s+\d+|GROUP\s+BY\b[^\n}]*|HAVING\b[^\n}]*)",
        re.IGNORECASE,
    )

    # Split into PREFIX+SELECT...WHERE{...} and the rest
    brace_depth = 0
    where_end = -1
    for i, ch in enumerate(query):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                where_end = i
                break

    if where_end > 0:
        inside = query[:where_end]
        after = query[where_end + 1:]

        # Extract misplaced clauses from inside WHERE {}
        extracted = clauses_inside_re.findall(inside)
        if extracted:
            inside = clauses_inside_re.sub("", inside).rstrip()
            # Remove trailing dots/whitespace
            inside = inside.rstrip(". \n\t")
            # Rebuild query
            suffix = " ".join(extracted) + " " + after.strip()
            query = inside + "\n}\n" + suffix.strip()
        else:
            query = inside + "\n}" + after

    # 2) Remove DESC/ASC without proper syntax (bare DESC after ORDER BY)
    query = re.sub(r"ORDER\s+BY\s+(\?\w+)\s+DESC\b", r"ORDER BY DESC(\1)", query, flags=re.IGNORECASE)
    query = re.sub(r"ORDER\s+BY\s+(\?\w+)\s+ASC\b", r"ORDER BY ASC(\1)", query, flags=re.IGNORECASE)

    # 3) Add LIMIT if not present
    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query = query.rstrip() + "\nLIMIT 20"

    return query.strip()


def generate_sparql(question: str, schema_summary: str, model: str = DEFAULT_MODEL) -> str:
    """Generate a SPARQL query from a natural language question."""
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question), model=model)
    query = extract_sparql_from_text(raw)
    query = sanitize_sparql(query)
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


def repair_sparql(
    schema_summary: str,
    question: str,
    bad_query: str,
    error_msg: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to fix a broken SPARQL query."""
    prompt = f"""The SPARQL query below has an error. Fix it and return only the corrected query in a ```sparql code block.
Use PREFIX aib: <http://example.org/ai-news/> for the local graph. Do NOT use SERVICE or wikibase.

Error: {error_msg}

Bad query:
{bad_query}

Question was: {question}

Corrected query:
"""
    raw = ask_local_llm(prompt, model=model)
    return sanitize_sparql(extract_sparql_from_text(raw))


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
