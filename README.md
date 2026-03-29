# AI-News Knowledge Graph

**Web Datamining & Semantics** — ESILV Project

End-to-end pipeline: web crawling, information extraction, knowledge base construction & alignment, SWRL reasoning, knowledge graph embeddings, and RAG (NL to SPARQL).

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd web-datamining-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Ollama (required for RAG module)
# macOS: brew install ollama
# Then pull a model:
ollama pull gemma:2b
```

## Project Structure

```
web-datamining-project/
├── src/
│   ├── crawl/          # Web crawler + cleaning
│   ├── ie/             # NER + relation extraction (spaCy)
│   ├── kg/             # KB construction, alignment, expansion
│   ├── reason/         # SWRL reasoning (OWLReady2)
│   ├── kge/            # Knowledge Graph Embeddings (PyKEEN)
│   └── rag/            # RAG pipeline (NL -> SPARQL via Ollama)
├── data/               # Generated data (CSVs, JSONs, models)
├── kg_artifacts/       # RDF files (ontology, alignment, expanded KB)
├── kge_datasets/       # Train/valid/test splits for KGE
├── reports/            # Experiment outputs (plots, CSVs)
├── notebooks/          # Jupyter notebooks
├── requirements.txt
└── .gitignore
```

## How to Run Each Module

All commands are run from the `src/` directory:

```bash
cd src
```

### Module 1 — Web Crawling & Information Extraction (3 pts)

Crawls 13 seed URLs (AI/tech domain), cleans HTML, extracts entities and relations.

```bash
python -m crawl.pipeline
```

**Outputs:** `data/crawler_output.jsonl`, `data/extracted_entities.csv`, `data/extracted_relations.csv`

### Module 2 — KB Construction, Alignment & Expansion (5 pts)

Builds RDF ontology, constructs initial KB, links entities/predicates to Wikidata, expands via SPARQL.

```bash
python -m kg.pipeline
```

Skip individual steps:
```bash
python -m kg.pipeline --skip-ontology --skip-build --skip-linking --skip-alignment
```

**Outputs:** `kg_artifacts/ontology.ttl`, `kg_artifacts/initial_kb.ttl`, `kg_artifacts/alignment.ttl`, `kg_artifacts/expanded.nt`

### Module 3 — SWRL Reasoning (part of 4 pts)

Runs SWRL rules with OWLReady2 + Pellet reasoner:
- **Family ontology:** `Person(?p) ^ hasAge(?p, ?age) ^ greaterThan(?age, 60) -> OldPerson(?p)`
- **Custom rule:** `Organization(?o) ^ develops(?o, ?p) ^ AIProduct(?p) -> AICompany(?o)`

```bash
python -m reason.pipeline
```

**Outputs:** Console output with inferred individuals, `data/swrl_inferred_results.csv`

### Module 4 — Knowledge Graph Embeddings (part of 4 pts)

Prepares data, trains TransE + ComplEx, evaluates on link prediction, runs experiments.

```bash
python -m kge.pipeline
```

Options:
```bash
python -m kge.pipeline --models TransE,ComplEx --epochs 100 --embedding-dim 200
python -m kge.pipeline --skip-train --skip-experiments  # only prepare + evaluate
```

**Outputs:** `kge_datasets/train.txt`, `data/kge_evaluation.csv`, `reports/tsne_*.png`, `reports/kb_size_sensitivity.csv`

### Module 5 — RAG Demo (4 pts)

NL to SPARQL generation with a local LLM (Ollama) 

**Prerequisites:** Ollama must be running with a model pulled:
```bash
ollama serve &       # start Ollama server
ollama pull gemma:2b # pull the model
```

Interactive mode (CLI chatbot):
```bash
python -m rag.pipeline
```

Batch evaluation (baseline vs RAG on 7 questions):
```bash
python -m rag.pipeline --evaluate
```

Options:
```bash
python -m rag.pipeline --model gemma:2b --kg-path kg_artifacts/expanded.nt
```

**Outputs:** `data/rag_evaluation.csv`
