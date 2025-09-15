# 07 Evaluation Pipeline.md

This document records the implementation of the **evaluation pipeline** used in `text2sql`.  
(Current pipeline includes **EM** and **ESM**; **ETM** has been extracted to a separate note.)

## Overview

1. Load `preds.txt` and `gold.txt`
2. Evaluate:
    - **EM (Exact Match)** — canonical SQL string match
    - **ESM (Execution Match)** — result-equivalence by executing SQL
3. Display and save results

---

## Inputs produced by the generation pipeline

Each run is saved under `run/` with a timestamped folder, e.g.:

`run/ └─ run_YYYYMMDD_HHMMSS/    ├─ gold.txt      # one SQL per line (gold)    ├─ preds.txt     # one SQL per line (prediction)    ├─ summary.jsonl # generation-time stats (latency, tokens, etc.)    └─ ...           # (other artifacts from generation)`

- For **unanswerable/error** queries, the SQL string in both `gold.txt` and `preds.txt` is the literal string `None`.

---

## How to run

From the repo root:

`python -m evalx.main`

### Auto-detect latest run

If `--run_dir` isn’t provided, the tool picks the most recent `run_*` folder by the timestamp in its name (falls back to latest mtime if no timestamped names are found).

### Useful flags

`python -m evalx.main \   --run_dir run/run_20250908_001148 \   --dsn postgresql://postgres:postgres@localhost:5433/t2sql_db \   --exec_timeout 30 \   --no_em   # (optional) skip EM   --no_esm  # (optional) skip ESM`

- `--dsn` default: `postgresql://postgres:postgres@localhost:5433/t2sql_db`
- `--exec_timeout` default: `30` seconds

---

## What gets written

Inside the run folder, an `eval/` directory is created:

`run/run_YYYYMMDD_HHMMSS/eval/ ├─ canon/ │  ├─ gold.txt   # canonicalized copy used for EM/ESM │  └─ preds.txt ├─ details.jsonl # per-example records └─ eval_summary.json`

- **Canonicalization** normalizes common formatting differences (e.g., extra spaces, trailing semicolons, etc.) before comparison or execution.
- **details.jsonl** contains one JSON record per example (index, gold SQL, pred SQL, and pass/fail booleans).
- **eval_summary.json** contains aggregated counts, rates, and ESM issue breakdown.

---

## Metrics

### 1) EM (Exact Match)

- Compares **canonicalized** `gold` and `pred` SQL **strings**.
- **Unanswerable rule:** if both `gold` and `pred` are `None`, EM counts as **pass** (model correctly decided the query is not executable).  
    If `gold` is `None` and `pred` is not, EM is a **fail**.

### 2) ESM (Execution Match)

- Executes `gold` and `pred` SQL against the database (via `--dsn`) and compares **result sets** for equivalence (order-insensitive where applicable).
- **Unanswerable rule:** if both `gold` and `pred` are `None`, ESM counts as **pass**.  
    If only one side is `None`, ESM is a **fail**.
- **Timeouts** and DB errors are recorded in the ESM breakdown (see below).

---

## Console output

You’ll see a compact summary like:

`=== EVAL SUMMARY === total: 30 empty_preds: 0 ; unanswerable: 1 EM  - 4/30 13.3% (of attempted: 4/30 13.3%) ESM - 27/30 90.0% (of attempted: 27/30 90.0%)  === ESM breakdown ===           ok: 27     mismatch: 3      skipped: 0   non_select: 0    gold_fail: 0    pred_fail: 0      timeout: 0  other_error: 0`

- **ok:** results matched
- **mismatch:** executed but results differed
- **non_select:** query not eligible for execution comparison
- **gold_fail/pred_fail:** DB error while running that side
- **timeout:** execution hit the configured timeout
- **other_error:** unexpected error category

---

## Inspecting ESM issues

Use the helper to print a breakdown and a few example cases:

`python scripts/inspect_eval.py --run_dir run/run_YYYYMMDD_HHMMSS --max_examples 5`

You’ll get per-bucket counts and (for sampled examples) the gold/pred SQL with the reason (e.g., mismatch).  
If you have a diff utility or notebook, you can also manually reproduce and inspect the result sets for those indices.

---

## Notes & assumptions

- The pipeline assumes your Postgres instance is reachable with the provided `--dsn`.
- Canonicalization is intentionally conservative—EM is about **exact** intent in string form; ESM checks **result equivalence**.
- ETM was removed from the core pipeline; see the separate **“ETM Notes”** document if you’d like to experiment with TreeMatch-style intent matching externally.