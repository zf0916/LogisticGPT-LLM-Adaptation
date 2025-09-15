# text2sql/scripts/run_generation.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import warnings
import os, sys, json, time, asyncio, traceback, logging, argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import sys
    
# Optional metrics
try:
    import psutil
except Exception:
    psutil = None
try:
    warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
    import pynvml
    _pynvml_ok = True
except Exception:
    _pynvml_ok = False
    pynvml = None

# Your vendor imports
from t2sql.agent import get_sql_agent
from t2sql.vectordb.chromadb import ChromaDB

# Our extensions
from extensions.agent_measure import MeasurableText2SQLAgent
from extensions.api import make_answer_ex

# --- Quiet down noisy libraries (tweak levels to taste) ---
logging.getLogger("t2sql").setLevel(logging.INFO)        # or WARNING
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("asyncpg").setLevel(logging.ERROR)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_lines(path: Path, lines: List[str]):
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def sql_one_line(sql: Optional[str]) -> str:
    if not sql:
        return ""
    s = sql.strip()
    # strip fenced code blocks if any
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n```$", "", s)
    # collapse all whitespace (including newlines/tabs) to single spaces
    s = re.sub(r"\s+", " ", s)
    # light cleanup around commas/parentheses
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()

class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.2):
        self.interval = interval_sec
        self.vram_peak_bytes = 0
        self.proc_rss_peak_bytes = 0
        self._stop = False

    def start(self):
        if _pynvml_ok:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass
        if psutil:
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None

    def sample(self):
        if self._proc:
            try:
                rss = self._proc.memory_info().rss
                self.proc_rss_peak_bytes = max(self.proc_rss_peak_bytes, rss)
            except Exception:
                pass
        if _pynvml_ok:
            try:
                n = pynvml.nvmlDeviceGetCount()
                for i in range(n):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.vram_peak_bytes = max(self.vram_peak_bytes, mem.used)
            except Exception:
                pass

    def stop(self):
        if _pynvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def vram_peak_gb(self) -> Optional[float]:
        return round(self.vram_peak_bytes / (1024 ** 3), 3) if self.vram_peak_bytes else None

    def rss_peak_gb(self) -> Optional[float]:
        return round(self.proc_rss_peak_bytes / (1024 ** 3), 3) if self.proc_rss_peak_bytes else None


@dataclass
class PerQueryResult:
    query_id: str
    db_id: str
    nl_query: str
    gold_sql: str
    pred_sql: Optional[str] = None
    gen_ok: Optional[bool] = None
    error: Optional[str] = None
    latency_ms_end_to_end: Optional[int] = None
    # Generation (model_sql)
    t2sql_tokens: Optional[int] = None
    t2sql_tokens_per_sec: Optional[float] = None
    # Retrieval / table selection (model_table_selection)
    retrieval_tokens: Optional[int] = None
    retrieval_tokens_per_sec: Optional[float] = None

@dataclass
class RunSummary:
    counts: Dict[str, int] = field(default_factory=lambda: {
        "total": 0,       # all queries attempted
        "ok": 0,          # gen_ok=True AND exec_ok=True
        "error": 0       # pipeline exceptions (tracebacks)
    })
    hardware: Dict[str, Any] = field(default_factory=dict)
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


async def build_agent(descriptor_base_path: str | None = None) -> MeasurableText2SQLAgent:
    """
    Use existing get_sql_agent(), then rewrap the underlying config/vector store
    into our measurable subclass without touching vendor code.
    """
    # create base agent first (validates config etc.)
    base_agent = get_sql_agent(descriptor_base_path)

    # reconstruct measurable agent using same config/vector store
    config = base_agent.config
    vector_store = base_agent._vector_store or ChromaDB(config)
    measurable = MeasurableText2SQLAgent(config, vector_store)

    # copy state we care about
    measurable._prompts = base_agent._prompts
    measurable._business_rules = base_agent._business_rules
    measurable._descriptor_folder = base_agent._descriptor_folder
    return measurable


async def run_generation():
    # ---- CLI args ----
    parser = argparse.ArgumentParser(description="Run Text-to-SQL generation")
    # Default: ETM OFF. Turn on with --etm. (Keep --no-etm for clarity/back-compat.)
    parser.add_argument("--etm", action="store_true",
                        help="Enable writing etm_preds.txt and etm_gold.txt (default: off)")
    parser.add_argument("--no-etm", action="store_true",
                        help="Force-disable ETM outputs (default)")
    args, _ = parser.parse_known_args()
    # precedence: --no-etm > --etm > default(False)
    if args.no_etm:
        write_etm = False
    elif args.etm:
        write_etm = True
    else:
        write_etm = False

    project_root = Path(__file__).resolve().parents[1]  # …/text2sql
    test_path = project_root / "data/annotate/test_set.jsonl"
    runs_root = project_root / "run"
    runs_root.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        print(f"[ERR] Test set not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    records = read_jsonl(test_path)
    clean = [r for r in records if all(k in r for k in ("query_id", "db_id", "nl_query", "gold_sql"))]
    if not clean:
        print("[ERR] No valid test entries.", file=sys.stderr)
        sys.exit(1)

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = runs_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_txt = out_dir / "preds.txt"
    gold_txt = out_dir / "gold.txt"
    if write_etm:
        etm_preds_txt = out_dir / "etm_preds.txt"
        etm_gold_txt  = out_dir / "etm_gold.txt"
    summary_path = out_dir / "summary.jsonl"

    preds_lines, gold_lines = [], []
    if write_etm:
        etm_preds_lines, etm_gold_lines = [], []
    
    monitor = ResourceMonitor()
    monitor.start()

    summary = RunSummary()
    summary.counts["total"] = len(clean)
    per_query: List[PerQueryResult] = []

    # Build agent once and reuse
    agent = await build_agent(descriptor_base_path=None)

    print(f"[INFO] Starting generation over {len(clean)} queries.")
    for idx, r in enumerate(clean, 1):
        qid, dbid, nlq, gsql = str(r["query_id"]), str(r["db_id"]), str(r["nl_query"]), str(r["gold_sql"])
        pq = PerQueryResult(query_id=qid, db_id=dbid, nl_query=nlq, gold_sql=gsql)

        t0 = time.time()
        error = None
        pred_sql = None
        gen_ok = None

        # --- baselines for per-query deltas (generation & retrieval buckets) ---
        base_gen_tokens = agent.usage_totals.get("generation", 0)
        base_tab_tokens = agent.usage_totals.get("table_selection", 0)
        base_gen_ms = agent.latency_totals_ms.get("generation", 0)
        base_tab_ms = agent.latency_totals_ms.get("table_selection", 0)

        try:
            # (1) Generate SQL (+ usage captured by measurable agent)
            sql, step, usage = await make_answer_ex(nlq, agent)
            pred_sql = sql
            gen_ok = bool(pred_sql and pred_sql.strip())

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        t1 = time.time()
        end_to_end_ms = int((t1 - t0) * 1000)

        # --- compute per-query deltas for generation & retrieval ---
        gen_tokens = agent.usage_totals.get("generation", 0) - base_gen_tokens
        tab_tokens = agent.usage_totals.get("table_selection", 0) - base_tab_tokens
        gen_ms = agent.latency_totals_ms.get("generation", 0) - base_gen_ms
        tab_ms = agent.latency_totals_ms.get("table_selection", 0) - base_tab_ms
        gen_tps = round(gen_tokens / (gen_ms / 1000.0), 4) if gen_ms > 0 and gen_tokens is not None else None
        tab_tps = round(tab_tokens / (tab_ms / 1000.0), 4) if tab_ms > 0 and tab_tokens is not None else None

        pq.pred_sql = pred_sql
        pq.gen_ok = gen_ok
        pq.error = error
        pq.latency_ms_end_to_end = end_to_end_ms
        # Generation metrics
        pq.t2sql_tokens = gen_tokens
        pq.t2sql_tokens_per_sec = gen_tps
        # Retrieval metrics
        pq.retrieval_tokens = tab_tokens
        pq.retrieval_tokens_per_sec = tab_tps
        per_query.append(pq)

        pred_out = "" if (error is not None or not gen_ok) else sql_one_line(pred_sql)
        preds_lines.append(pred_out)
        gold_lines.append(gsql)  # gold already comes as "None" string if null in JSONL
        if write_etm:
            etm_preds_lines.append(f"{pred_out}\t{dbid}")
            etm_gold_lines.append(f"{gsql}\t{dbid}")
        summary.counts["ok"] += 1 if (error is None and gen_ok) else 0
        summary.counts["error"] += 1 if (error is not None) else 0

        monitor.sample()
        # Keep status "OK" for non-crash paths; fine-grained buckets are printed anyway
        status = "OK" if error is None else "ERR"
        print(
            f"[{idx}/{len(clean)}] {qid} | {status}"
            f" | end2end={end_to_end_ms} ms"
            f" | gen_tokens={gen_tokens} gen_tps={gen_tps}"
            f" | retr_tokens={tab_tokens} retr_tps={tab_tps}"
        )
    
    monitor.stop()

    lat = [p.latency_ms_end_to_end for p in per_query if p.latency_ms_end_to_end is not None]
    gen_tps_vals  = [p.t2sql_tokens_per_sec for p in per_query if p.t2sql_tokens_per_sec is not None]
    retr_tps_vals = [p.retrieval_tokens_per_sec for p in per_query if p.retrieval_tokens_per_sec is not None]
    gen_tps_avg   = round(sum(gen_tps_vals)/len(gen_tps_vals), 4) if gen_tps_vals else None
    retr_tps_avg  = round(sum(retr_tps_vals)/len(retr_tps_vals), 4) if retr_tps_vals else None 
    
    summary.hardware = {
        "latency_ms_avg": int(sum(lat)/len(lat)) if lat else None,
        "vram_peak_gb_max": monitor.vram_peak_gb(),
        "proc_rss_peak_gb": monitor.rss_peak_gb(),
        "generation_tokens_total": agent.usage_totals.get("generation"),
        "generation_tokens_per_sec_avg": gen_tps_avg,
        "retrieval_tokens_total": agent.usage_totals.get("table_selection"),
        "retrieval_tokens_per_sec_avg": retr_tps_avg,
    }

    write_lines(preds_txt, preds_lines)
    write_lines(gold_txt, gold_lines)
    if write_etm:
        write_lines(etm_preds_txt, etm_preds_lines)
        write_lines(etm_gold_txt, etm_gold_lines)
    else:
        print("[INFO] ETM outputs disabled (--no-etm).")
    summary_path.write_text(summary.to_json() + "\n", encoding="utf-8")

    print("\n=== RUN SUMMARY ===")
    print(summary.to_json())
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    asyncio.run(run_generation())

# Run script:
# python scripts/run_generation.py