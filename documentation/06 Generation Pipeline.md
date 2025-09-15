# 06 Generation Pipeline

This document describes the batch **Text-to-SQL generation pipeline** used to run your curated test set through the Datrics agent, collect predictions, and log hardware/usage metrics. It reflects the final script and light-weight extensions you added.

---

## Overview

* **Entry point:** `text2sql/scripts/run_generation.py`
* **Agent:** `MeasurableText2SQLAgent` (extension wrapper) built from `t2sql.agent.get_sql_agent()`
* **Execution:** Async loop over a JSONL test set → generate SQL → optionally execute → write outputs → one-line JSON summary.
* **No Streamlit required.** None of the modules used here depend on `st.session_state`.

### Minimal dependencies (optional)

```bash
pip install psutil nvidia-ml-py
```

> Use `nvidia-ml-py` (which exposes the `pynvml` module) to avoid the deprecation warning. VRAM metrics become `null` if no NVIDIA GPU.

---

## Test Set Loading

1. **Path:** `data/annotate/test_set.jsonl`
2. **Format (per line):**

   ```json
   {"query_id": "Q123", "db_id": "pickup", "nl_query": "...", "gold_sql": "..."}
   ```
3. The script performs light validation (skips malformed rows, continues without crashing).

---

## Agent Initialization & Calls

* **Build once, reuse:**

  * `agent = await build_agent()` creates a base `Text2SQLAgent` via `get_sql_agent(...)` and wraps it in `MeasurableText2SQLAgent`.
* **Calls per query:**

  1. **Generate SQL** → `make_answer_ex(query, agent)` returns `(sql, step, usage_dict)`
  2. **Execute SQL** → `run_sql_ex(sql, agent)` returns `(sql, success: bool, data)`

### Success flags

* **`gen_ok`** — True if generated SQL is non-empty.
* **`exec_ok`** — True if DB execution returns without error (after optional auto-fix/timeout logic).
* **Rows preview** is only recorded when `exec_ok=True`.

### Auto-fix, timeouts & quiet logs

* The measurable agent overrides `execute_sql` to:

  * **Timeout** each DB call (default `sql_exec_timeout_s = 30`).
  * **Auto-fix** SQL errors **except** for fatal schema errors (by default skips auto-fix on `undefined_column` / `undefined_table`).
  * **Quiet** error spam (single-line warnings instead of full stack traces).

**Config knobs** (set in descriptor JSON if desired):

```json
{
  "quiet_sql_errors": true,
  "max_sql_fix_attempts": 1,
  "sql_exec_timeout_s": 30,
  "sql_autofix_enabled": true,
  "sql_autofix_blacklist": ["undefined_column", "undefined_table"]
}
```

### Unanswerable detection

* If execution fails with `undefined_column` or `undefined_table`, the pipeline marks the query as **`unanswerable=True`** and **does not** return rows.

---

## SQL Prompting Rules (descriptor)

The default instruction string (one line with escaped newlines) was updated to bias away from slow anti-joins and to standardize SQL style:

```python
DEFAULT_SQL_INSTRUCTIONS = "Mandatory use these INSTRUCTIONS in Chain-of-Thoughts:\n1. Minimize the number of tables used in each query and avoid unnecessary joins or operations.\n2. Before applying aggregations (e.g., SUM, AVG), ensure empty or invalid values are excluded.\n3. Qualify tables as public.pickup p and public.delivery d, and use aliases p and d consistently.\n4. Use COUNT(*), explicit JOIN ... ON ... clauses, UPPERCASE SQL keywords with lower_snake_case identifiers.\n5. Express durations as EXTRACT(EPOCH FROM (end_ts - start_ts)) / 3600.0.\n6. Prefer LEFT JOIN ... IS NULL for anti-joins instead of NOT IN (SELECT ...)."
```

> Tip: If you see long runtimes on anti-joins, also add indexes:
>
> ```sql
> CREATE INDEX IF NOT EXISTS idx_pickup_order_id   ON public.pickup(order_id);
> CREATE INDEX IF NOT EXISTS idx_delivery_order_id ON public.delivery(order_id);
> ```

---

## Hardware & Usage Metrics

For each query, the agent tracks two buckets:

* **Generation** (final SQL write; `model_sql`)
* **Retrieval / Table Selection** (`model_table_selection`)

### Per-query fields (subset)

* `latency_ms_end_to_end` — NL → SQL → (optional) Exec
* `t2sql_tokens` & `t2sql_tokens_per_sec` — **generation** tokens and rate
* `retrieval_tokens` & `retrieval_tokens_per_sec` — **table-selection** tokens and rate

Runtime samplers:

* **VRAM peak (GB):** via `nvidia-ml-py` (exposes `pynvml`) if GPU present
* **Process RSS peak (GB):** via `psutil`

---

## Outputs

All artifacts go to `run/<run_id>/` (e.g., `run/run_20250907_014000/`):

* `preds.txt` — one `pred_sql` per line (empty string if none)
* `gold.txt` — one `gold_sql` per line
* `etm_preds.txt` — `pred_sql + "\t" + db_id`
* `etm_gold.txt` — `gold_sql + "\t" + db_id`
* `summary.jsonl` — a single JSON line with counts + hardware

### Summary schema (final)

```json
{
  "counts": {
    "total": 500,
    "ok": 496,            // gen_ok=True AND exec_ok=True
    "error": 2,           // pipeline exceptions
    "exec_fail": 1,       // non-fatal execution failures
    "unanswerable": 1     // missing column/table
  },
  "hardware": {
    "latency_ms_avg": 895,
    "vram_peak_gb_max": 12.4,
    "proc_rss_peak_gb": 3.12,

    "generation_tokens_total": 123456,
    "generation_tokens_per_sec_avg": 321.0,

    "retrieval_tokens_total": 7890,
    "retrieval_tokens_per_sec_avg": 110.5,
  }
}
```

---

## CLI Usage

From project root:

```bash
# Windows PowerShell (venv optional)
# .\.venv\Scripts\Activate.ps1
python text2sql/scripts/run_generation.py
```

---

## Known Behaviors & Troubleshooting

* **Missing schema fields** → flagged as `unanswerable=True` (no auto-fix attempted; no rows returned).
* **Long-running queries** → bounded by `sql_exec_timeout_s` (default 30s). Prefer **LEFT JOIN ... IS NULL** over `NOT IN (SELECT ...)` and add the order\_id indexes if needed.
* **VRAM metric warning** → use `nvidia-ml-py` instead of `pynvml`.
* **Noise in logs** → logging levels are tightened in the script; adjust if you need more/less detail.

---
