from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

from .normalization import normalize_rows

# Optional DB drivers
try:
    import psycopg2  # type: ignore
    _HAS_PG2 = True
except Exception:
    _HAS_PG2 = False

try:
    import psycopg  # type: ignore  # psycopg3
    _HAS_PG3 = True
except Exception:
    _HAS_PG3 = False


@dataclass
class LineEval:
    idx: int
    pred_sql: str
    gold_sql: str
    em: Optional[bool] = None
    esm: Optional[bool] = None
    etm: Optional[bool] = None
    error: Optional[str] = None


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def load_etm_lines(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in load_lines(path):
        if "\t" in line:
            sql, dbid = line.split("\t", 1)
        else:
            sql, dbid = line, ""
        out.append((sql, dbid))
    return out


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_select_only(sql: str) -> bool:
    s = (sql or "").strip().lower()
    if not s:
        return False
    # Accept plain SELECT and CTEs starting with WITH
    return s.startswith("select") or s.startswith("with")

def _pg_connect(dsn: str):
    if _HAS_PG2:
        return psycopg2.connect(dsn)
    if _HAS_PG3:
        return psycopg.connect(dsn)
    raise RuntimeError("No psycopg2/psycopg driver installed; cannot run ESM.")


def _set_timeout(cur, timeout_ms: int):
    try:
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
    except Exception:
        pass  # best‑effort; not critical


def exec_query(dsn: str, sql: str, limit: int = 10000, timeout_s: int = 30) -> Tuple[bool, list[tuple], Optional[str]]:
    # Trim trailing semicolons to keep subquery wrapper valid
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not safe_select_only(cleaned):
        return False, [], "Non-SELECT or empty SQL"
    wrapper = f"SELECT * FROM ({cleaned}) AS _t LIMIT {int(limit)}"
    try:
        conn = _pg_connect(dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    _set_timeout(cur, max(1000, int(timeout_s * 1000)))
                    t0 = time.time()
                    cur.execute(wrapper)
                    rows = cur.fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True, rows, None
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"


def rows_equal_unordered(a: List[Tuple], b: List[Tuple]) -> bool:
    na = normalize_rows(a)
    nb = normalize_rows(b)
    if na == nb:
        return True

    # Relaxed singleton rule:
    # If both have exactly 1 row, and one side is 1 column while the other has >=1 columns,
    # consider equal if the single normalized value is present in the other row.
    if len(na) == 1 and len(nb) == 1:
        ra, rb = na[0], nb[0]
        if len(ra) == 1 and len(rb) >= 1:
            return ra[0] in rb
        if len(rb) == 1 and len(ra) >= 1:
            return rb[0] in ra

    return False

def _is_unanswerable_sql(s: str | None) -> bool:
    t = (s or "").strip().lower()
    return t in ("", "none", "null")

def esm_bucket(d: "LineEval") -> str:
    err = (d.error or "").strip().lower()
    if not err and d.esm is True:
        return "ok"
    if "non-select" in err:
        return "non_select"
    if "gold exec fail" in err:
        return "gold_fail"
    if "pred exec fail" in err:
        return "pred_fail"
    if "timeout" in err:
        return "timeout"
    if d.esm is False:
        return "mismatch"
    if d.esm is None:
        return "skipped"
    return "other_error"

def summarize_detail(details: list[LineEval]) -> dict:
    total = len(details)

    em_pass = sum(d.em  is True  for d in details)
    em_fail = sum(d.em  is False for d in details)
    esm_pass = sum(d.esm is True  for d in details)
    esm_fail = sum(d.esm is False for d in details)
    etm_pass = sum(d.etm is True  for d in details)
    etm_fail = sum(d.etm is False for d in details)

    def _pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 1) if den else 0.0

    summary = {
        "total": total,
        "counts": {
            "em":  {"pass": em_pass,  "fail": em_fail},
            "esm": {"pass": esm_pass, "fail": esm_fail},
            "etm": {"pass": etm_pass, "fail": etm_fail},
        },
        "skips": {
            "empty_pred":   sum(1 for d in details if (d.pred_sql or "").strip() == ""),
            "unanswerable": sum(1 for d in details if _is_unanswerable_sql(d.gold_sql)),
        },
        "errors": sum(
            1 for d in details
            if (d.error or "").strip() and not _is_unanswerable_sql(d.gold_sql)
        ),
        # NEW: percentages (both vs total and vs attempted)
        "rates": {
            "em": {
                "attempted": em_pass + em_fail,
                "pass_pct_total": _pct(em_pass, total),
                "pass_pct_attempted": _pct(em_pass, em_pass + em_fail),
            },
            "esm": {
                "attempted": esm_pass + esm_fail,
                "pass_pct_total": _pct(esm_pass, total),
                "pass_pct_attempted": _pct(esm_pass, esm_pass + esm_fail),
            },
            "etm": {
                "attempted": etm_pass + etm_fail,
                "pass_pct_total": _pct(etm_pass, total),
                "pass_pct_attempted": _pct(etm_pass, etm_pass + etm_fail),
            },
        },
    }

    # (Optional) ESM breakdown you added earlier
    buckets = {"ok":0,"mismatch":0,"skipped":0,"non_select":0,"gold_fail":0,"pred_fail":0,"timeout":0,"other_error":0}
    for d in details:
        err = (d.error or "").strip().lower()
        if not err and d.esm is True:
            buckets["ok"] += 1
        elif "non-select" in err:
            buckets["non_select"] += 1
        elif "gold exec fail" in err:
            buckets["gold_fail"] += 1
        elif "pred exec fail" in err:
            buckets["pred_fail"] += 1
        elif "timeout" in err:
            buckets["timeout"] += 1
        elif d.esm is False:
            buckets["mismatch"] += 1
        elif d.esm is None:
            buckets["skipped"] += 1
        else:
            buckets["other_error"] += 1
    summary["esm_breakdown"] = buckets

    return summary

def to_records(details: list[LineEval]) -> list[dict]:
    return [asdict(d) for d in details]