# scripts/make_etm_schema.py
from __future__ import annotations
import json
from pathlib import Path

def spider_type(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ["timestamp", "time_window", "accept_ts"]):
        return "time"
    if c.endswith("_id") or c in {"lng","lat","cnt","count","deliveries","pickups"}:
        return "number"
    if c in {"region_id","courier_id","aoi_id"}:
        return "number"
    return "text"

def build_schema():
    db_id = "t2sql"
    tables = [
        ("pickup", [
            "order_id","region_id","city","courier_id","lng","lat",
            "aoi_id","aoi_type","timestamp_utc","accept_ts_utc",
            "time_window_start_utc","time_window_end_utc"
        ]),
        ("delivery", [
            "order_id","region_id","city","courier_id","lng","lat",
            "aoi_id","aoi_type","timestamp_utc","accept_ts_utc"
        ]),
    ]
    table_names = [t for t,_ in tables]
    column_names = [[-1,"*"]]
    column_names_original = [[-1,"*"]]
    column_types = [""]

    for ti, (_, cols) in enumerate(tables):
        for c in cols:
            column_names.append([ti, c])
            column_names_original.append([ti, c])
            column_types.append(spider_type(c))

    return [{
        "db_id": db_id,
        "table_names": table_names,
        "table_names_original": table_names[:],
        "column_names": column_names,
        "column_names_original": column_names_original,
        "column_types": column_types,     # Spider coarse types
        "primary_keys": [],
        "foreign_keys": []
    }]

def main():
    out = Path("ETM/etm_db/tables.json")  # adjust if you like
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_schema(), indent=2), encoding="utf-8")
    print(f"[make_etm_schema] wrote {out}")

if __name__ == "__main__":
    main()
