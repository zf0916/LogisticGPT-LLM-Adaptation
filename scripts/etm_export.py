# scripts/etm_export.py
from __future__ import annotations
from pathlib import Path
import re, argparse

TAB_DB_SEP = "\t"

def is_unans(s: str | None) -> bool:
    return (s or "").strip().lower() in ("", "none", "null")

def oneline(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r";\s*$", "", s)
    return s

# ----- Dialect: PG -> MySQL-ish that ETM likes -----
def strip_schema(s: str) -> str:
    return re.sub(r"\bpublic\.", "", s, flags=re.IGNORECASE)

def drop_pg_casts(s: str) -> str:
    s = re.sub(r"::\s*double\s+precision\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"::\s*(bigint|int|integer|smallint|numeric|decimal|real|float|date|timestamp|timestamptz|time|boolean)\b", "", s, flags=re.IGNORECASE)
    return s

def drop_nulls_order(s: str) -> str:
    return re.sub(r"\s+NULLS\s+(FIRST|LAST)\b", "", s, flags=re.IGNORECASE)

def normalize_interval_literal(s: str) -> str:
    return re.sub(r"\bINTERVAL\s+'([^']*)'", r"'\1'", s, flags=re.IGNORECASE)

def pg_to_mysqlish(s: str) -> str:
    s = re.sub(r"DATE_TRUNC\s*\(\s*'day'\s*,\s*([^)]+?)\)", r"DATE(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"EXTRACT\s*\(\s*HOUR\s+FROM\s+([^)]+?)\)", r"HOUR(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"EXTRACT\s*\(\s*DOW\s+FROM\s+([^)]+?)\)", r"DAYOFWEEK(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"EXTRACT\s*\(\s*EPOCH\s+FROM\s*\(\s*([^)]+?)\s*-\s*([^)]+?)\s*\)\s*\)\s*/\s*3600(?:\.0)?\b",
               r"TIMESTAMPDIFF(HOUR, \2, \1)", s, flags=re.IGNORECASE)
    return s

# ----- NEW: alias -> base table rewrite (preserve qualifiers) -----
ALLOWED_BASE = {"delivery", "pickup"}

def build_alias_map(s: str) -> dict[str,str]:
    amap = {}
    # FROM <table> <alias>
    for m in re.finditer(r"\bFROM\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)", s, flags=re.IGNORECASE):
        tbl, alias = m.group(1).lower(), m.group(2)
        if tbl in ALLOWED_BASE:
            amap[alias] = tbl
    # JOIN <table> <alias>
    for m in re.finditer(r"\bJOIN\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)", s, flags=re.IGNORECASE):
        tbl, alias = m.group(1).lower(), m.group(2)
        if tbl in ALLOWED_BASE:
            amap[alias] = tbl
    return amap

def replace_alias_qualifiers(s: str, amap: dict[str,str]) -> str:
    # d.timestamp_utc -> delivery.timestamp_utc ; p.order_id -> pickup.order_id
    for alias, base in sorted(amap.items(), key=lambda kv: -len(kv[0])):  # longest first
        s = re.sub(rf"\b{re.escape(alias)}\.", base + ".", s)
    return s

def drop_base_alias_tokens(s: str, amap: dict[str,str]) -> str:
    # FROM delivery d -> FROM delivery ; JOIN pickup p -> JOIN pickup
    for alias, base in amap.items():
        s = re.sub(rf"\bFROM\s+{base}\s+{re.escape(alias)}\b", f"FROM {base}", s, flags=re.IGNORECASE)
        s = re.sub(rf"\bJOIN\s+{base}\s+{re.escape(alias)}\b", f"JOIN {base}", s, flags=re.IGNORECASE)
    return s

# keep qualifiers (delivery./pickup.) inside function args; do NOT strip them

# ----- Identifier unification (safe renames) -----
def canonical_identifiers(s: str) -> str:
    # counters
    s = re.sub(r"\b(daily_count|delivery_count|pickup_count|deliveries|pickups)\b", "cnt", s, flags=re.IGNORECASE)
    # date bucket
    s = re.sub(r"\b(delivery_date|pickup_date)\b", "day", s, flags=re.IGNORECASE)
    # hour bucket
    s = re.sub(r"\bpickup_hour\b", "hour_of_day", s, flags=re.IGNORECASE)
    # generic duration name (only aliases, not TIMESTAMPDIFF itself)
    s = re.sub(r"\b(avg_delivery_time_hours|avg_pickup_delay_hours|cycle_time_hours|delivery_time_hours|hours_between_pickup_and_delivery)\b",
               "hours", s, flags=re.IGNORECASE)
    return s

def normalize_for_etm(sql: str) -> str:
    s = oneline(sql)
    s = strip_schema(s)
    s = drop_pg_casts(s)
    s = drop_nulls_order(s)
    s = normalize_interval_literal(s)
    s = pg_to_mysqlish(s)
    amap = build_alias_map(s)
    s = replace_alias_qualifiers(s, amap)
    s = drop_base_alias_tokens(s, amap)
    s = canonical_identifiers(s)
    return re.sub(r"\s+", " ", s).strip()

def load_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--db_id", default="t2sql")
    ap.add_argument("--out_gold", default=None)
    ap.add_argument("--out_pred", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    gold = rd / "eval" / "canon" / "gold.txt"
    pred = rd / "eval" / "canon" / "preds.txt"
    if not gold.exists(): gold = rd / "gold.txt"
    if not pred.exists(): pred = rd / "preds.txt"

    g_lines = load_lines(gold)
    p_lines = load_lines(pred)
    n = min(len(g_lines), len(p_lines))

    gold_out = Path(args.out_gold or (rd / "etm_gold.txt"))
    pred_out = Path(args.out_pred or (rd / "etm_preds.txt"))

    out_g, out_p, skipped = [], [], 0
    for i in range(n):
        g, p = g_lines[i], p_lines[i]
        if is_unans(g) or is_unans(p):
            skipped += 1
            continue
        g1 = normalize_for_etm(g)
        p1 = normalize_for_etm(p)
        out_g.append(f"{g1}{TAB_DB_SEP}{args.db_id}")
        out_p.append(f"{p1}{TAB_DB_SEP}{args.db_id}")

    gold_out.write_text("\n".join(out_g) + ("\n" if out_g else ""), encoding="utf-8")
    pred_out.write_text("\n".join(out_p) + ("\n" if out_p else ""), encoding="utf-8")
    print(f"[etm_export] wrote {len(out_g)} pairs (skipped {skipped})")
    print(f"  gold: {gold_out}")
    print(f"  pred: {pred_out}")

if __name__ == "__main__":
    main()
