LogisticGPT: Text-to-SQL for Last-Mile Logistics on Consumer-Grade Hardware

Goal. Adapt and evaluate a Text-to-SQL pipeline for logistics (pickup & delivery analytics) that runs reliably on a CPU or ≤16 GB VRAM GPU, with rigorous, execution-based evaluation and canonical SQL outputs.

This repository contains my research code, configs, and docs for building a logistics-focused Text-to-SQL system powered by a RAG (Retrieval-Augmented Generation) pipeline and evaluated on tasks grounded in the Cainiao LaDe last-mile delivery dataset. The project uses the open-source Datrics Text2SQL pipeline as the generation backbone and adds domain-specific canonicalization, sanitization, and evaluation layers.

RAG-powered Text-to-SQL built on top of the Datrics Text2SQL engine (Streamlit app optional). You can swap between local and hosted LLMs.

Logistics domain focus using pickup and delivery tables with real last-mile analytics questions.

Consumer-grade hardware target: designed to work with ≤16 GB VRAM, tracking tokens/sec and latency.

Strict canonical SQL with schema qualifiers (public.pickup p, public.delivery d), safe casts, and execution-ready outputs.

Evaluation suite: EM (exact string), ESM (execution match), ETM (intent/tree match) + inspection tools.

Chroma vector index with OpenAI embeddings by default (configurable).

Architecture
Natural Language Question
          │
          ▼
   Retrieval (Chroma, docs, DDL, examples)
          │
          ▼
  Datrics Text2SQL Generator  ──>  SQL
          │                         │
          ├── SQL Sanitizer & Canonicalizer (this repo)
          │                         │
          ▼                         ▼
     Postgres (exec)         Evaluation (EM / ESM / ETM)
                                    │
                                    ▼
                           Reports & Inspect Tools


Separation of concerns

This repo: dataset prep, configuration, canonicalization/sanitization, evaluation, scripts, and documentation for logistics.

Datrics Text2SQL: upstream generation pipeline and Streamlit UI. We integrate it rather than re-implement it. 
GitHub

Getting Started
Prerequisites

Python 3.12+

Docker (for Postgres) or a local Postgres 14/15 instance

GPU (optional): ≤16 GB VRAM if running local models

OpenAI API key (if using OpenAI embeddings / models)

Installation
# Clone your repo
git clone https://github.com/<you>/<repo>.git
cd <repo>

# (Recommended) Create venv
python -m venv .venv
./.venv/Scripts/activate   # Windows
# source .venv/bin/activate # macOS/Linux

# Install
pip install -r requirements.txt

Quickstart
# 1) Start Postgres (docker example)
docker run --name t2sql_db -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 -d postgres:15

# 2) Create schema & tables (pickup, delivery)
psql "postgresql://postgres:postgres@localhost:5433/postgres" -f sql/schema.sql

# 3) Load prepared CSVs
psql "postgresql://postgres:postgres@localhost:5433/postgres" -f sql/load_data.sql

# 4) Ingest documentation/DDL/examples into Chroma
python scripts/ingest_knowledge.py --config config/descriptor.json

# 5) Run generation on a query file and save outputs
python scripts/generate_sql.py --input data/queries.jsonl --out run/latest/predictions.jsonl

# 6) Evaluate EM/ESM/ETM
python scripts/evaluate.py --run_dir run/latest
python scripts/inspect_eval.py --run_dir run/latest

Configuration

All key settings live in config/descriptor.json (example):

{
  "db": {
    "uri": "postgresql://postgres:postgres@localhost:5433/postgres",
    "schema": "public"
  },
  "vector_store": {
    "provider": "chroma",
    "db_path": "index/chroma"
  },
  "embeddings": {
    "backend": "openai",
    "model": "text-embedding-3-small",
    "env_var": "OPENAI_API_KEY"
  },
  "generation": {
    "router_model_list": [
      { "model_name": "sqlcoder-local", "litellm_params": { "model": "ollama/sqlcoder:7b" } },
      { "model_name": "gpt-4o-mini",     "litellm_params": { "model": "gpt-4o-mini" } }
    ]
  },
  "retrieval": {
    "n_results_sql": 8,
    "n_results_documentation": 8,
    "n_results_ddl": 8
  }
}


Note: By design, this project defaults to OpenAI embeddings for the Chroma index. You can switch to an open-source embedding model if needed, but the default is chosen for stability in this pipeline.

Set environment variables:

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."

Data Setup

This project expects a pickup and delivery table (LaDe-inspired). We reference Cainiao LaDe for terminology and task framing:

LaDe is a large-scale last-mile dataset with millions of packages and rich event logs (e.g., task accept/finish timestamps). 
GitHub
+2
arXiv
+2

The original LaDe resources are available via GitHub, arXiv, and Hugging Face (see references). Use them to understand schema patterns and tasks; this repo does not redistribute LaDe. 
GitHub
+2
arXiv
+2

Local tables we use

public.pickup (e.g., order_id, region_id, city, courier_id, lng, lat, aoi_id, aoi_type, timestamp_utc, accept_ts_utc)

public.delivery (same timestamp fields in _utc)

CSV load scripts are provided in sql/.

Evaluation

We report:

EM (Exact Match): predicted SQL string equals gold SQL.

ESM (Execution Match): both queries execute and results are identical.

ETM (Enhanced Tree/Intent Match): relaxed structural/intent equivalence.

How to run

python scripts/evaluate.py --run_dir run/run_YYYYMMDD_HHMMSS
python scripts/inspect_eval.py --run_dir run/run_YYYYMMDD_HHMMSS --max_examples 30


Artifacts:

predictions.jsonl, preds.txt, gold.txt

summary.jsonl with metric breakdowns

Example diffs and failing cases for rapid iteration

Canonical SQL Style

To maximize execution reliability and comparability:

Always qualify schema & alias:

public.pickup p, public.delivery d

Use explicit aggregates:

COUNT(*) AS total, AVG(...)

Time handling:

Grouping by date: DATE_TRUNC('day', d.timestamp_utc) AS day

Hours from seconds: EXTRACT(EPOCH FROM (d.timestamp_utc - d.accept_ts_utc)) / 3600.0

Avoid ambiguous SELECT * in final answers (except for debugging).

Order deterministic outputs: ORDER BY ... ASC.

Project Structure
.
├─ config/
│  └─ descriptor.json         # DB, embeddings, retrieval, model router
├─ data/
│  ├─ queries.jsonl           # NLQ + gold SQL (examples)
│  └─ processed/              # your prepared CSVs (not committed)
├─ index/
│  └─ chroma/                 # Chroma persistent index
├─ run/
│  └─ run_YYYYMMDD_HHMMSS/    # per-run outputs (predictions, metrics)
├─ scripts/
│  ├─ ingest_knowledge.py     # ingest docs/DDL/examples into Chroma
│  ├─ generate_sql.py         # call Datrics pipeline, produce predictions
│  ├─ sanitize_sql.py         # fix minor issues (casts, schema, case)
│  ├─ canonicalize_sql.py     # enforce house SQL style
│  ├─ evaluate.py             # EM/ESM/ETM evaluation
│  └─ inspect_eval.py         # failure analysis
├─ sql/
│  ├─ schema.sql              # create pickup/delivery tables
│  └─ load_data.sql           # load CSVs into tables
└─ README.md

Results & Reproducibility

Each experiment is stored under run/run_YYYYMMDD_HHMMSS with:

Model/router config snapshot

Timing and tokens/sec

EM/ESM/ETM summaries

Problematic cases for inspection

To reproduce:

Use the same config/descriptor.json.

Re-ingest with the same OpenAI embedding model (or pin your OSS model).

Re-run generate_sql.py and evaluate.py on the same data/queries.jsonl.

Roadmap

🔁 Expand query set for broader logistics intents (SLA breaches, regional trends).

🧪 Add unit tests for sanitizer/canonicalizer rules.

📊 Auto-report notebooks for run comparisons.

🧩 Optional local embedding model preset (fallback when OpenAI is unavailable).

Acknowledgements & References

Cainiao LaDe Dataset
GitHub: LaDe — code & dataset pointers (first publicly available last-mile delivery dataset with millions of packages). 
[GitHub](https://github.com/wenhaomin/LaDe?utm_source=chatgpt.com)

Paper (arXiv): LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry. 
[arXiv](https://arxiv.org/pdf/2306.10675?utm_source=chatgpt.com)

Hugging Face dataset card: overview of scale and fields. 
[Hugging Face](https://huggingface.co/datasets/Cainiao-AI/LaDe?utm_source=chatgpt.com)

Datrics Text2SQL
GitHub: Text2SQL Engine with advanced RAG (Apache-2.0). 
[GitHub](https://github.com/datrics-ai/text2sql?utm_source=chatgpt.com)

Preprint: Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation. 
https://arxiv.org/html/2506.12234v1?utm_source=chatgpt.com

This repo uses Datrics’ generation pipeline; it does not re-license or redistribute Datrics code. Please consult their repo and license for usage terms.