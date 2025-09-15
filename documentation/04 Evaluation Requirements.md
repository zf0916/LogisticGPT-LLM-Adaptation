04 Evaluation Requirements.md

Hardware Metrics
1. Latency: End-to-end response time per query (NL input → SQL output)
2. Tokens per second: Required >= 1 token/sec
3. VRAM usage: Required < 16 GB VRAM (for local models)
4. CPU/Memory

Result Metrics
1. EM (Exact Match) - Same SQL string
2. ESM (Execution match) - Same SQL execution result
3. ETM (Enhanced Tree Matching) - Same intent
- Refer paper: https://arxiv.org/pdf/2407.07313
- Refer github: https://github.com/emorynlp/ETM/

Preds and Gold Format
1. preds.txt: stream pred_sql
2. gold.txt: stream gold_sql
3. etm_preds.txt: write pred_sql + "\t" + db_id
4. etm_gold.txt: write gold_sql + "\t" + db_id

Run format
1. per query (predictions.jsonl):
{
  "query_id": "q_000123",
  "db_id": "pickup",
  "nl_query": "How many pickups were on 2025-01-12 in Shanghai?",
  "gold_sql": "SELECT COUNT(*) FROM pickup WHERE city='Shanghai' AND DATE(timestamp_utc)='2025-01-12';",
  "pred_sql": "SELECT COUNT(*) FROM pickup WHERE city='Shanghai' AND DATE(timestamp_utc)='2025-01-12';",

  "hardware": {
    "latency_ms": 842,
    "tokens_per_sec": 1.23,
    "vram_peak_gb": 9.8
  },

  "results": {
    "em": true,
    "esm": true,
    "etm": true
  }
}

2. summary (summary.json)
{
  "counts": {
    "total": 500,
    "ok": 496,
    "error": 4
  },
  "metrics": {
    "em": { "acc": 0.842 },
    "esm": { "acc": 0.904 },
    "etm": { "acc": 0.918 }
  },
  "hardware": {
    "latency_ms_avg": 895,
    "tokens_per_sec_avg": 1.28,
    "vram_peak_gb_max": 12.4
  }
}
