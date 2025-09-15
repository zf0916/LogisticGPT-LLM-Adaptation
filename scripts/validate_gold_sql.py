#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

def env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)

def connect():
    params = dict(
        host=env("PGHOST", "localhost"),
        port=int(env("PGPORT", "5433")),
        dbname=env("PGDATABASE", "t2sql_db"),
        user=env("PGUSER", "postgres"),
        password=env("PGPASSWORD", "postgres"),
    )
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        print(f"[FATAL] Could not connect to Postgres with params {params}: {e}", file=sys.stderr)
        sys.exit(2)
    return conn

def validate_jsonl(path: str, timeout_ms: int = 60000, limit_preview: int = 3) -> int:
    conn = connect()
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=RealDictCursor)
    # Set per-statement timeout (ms)
    cur.execute(f"SET statement_timeout = {timeout_ms};")
    passed = 0
    failed = 0
    total = 0
    print(f"✓ Connected. Using statement_timeout={timeout_ms} ms\n")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[{total:03d}] ❌ JSON parse error: {e}\n  line={line[:200]}...", flush=True)
                failed += 1
                continue

            qid = obj.get("query_id") or obj.get("id") or f"row_{total}"
            sql = obj.get("gold_sql")
            if sql is None:
                print(f"[{total:03d}] {qid}: ⚠️  gold_sql is null (unanswerable) — skipped.")
                continue

            # Trim and ensure a terminal semicolon-friendly execution
            sql_clean = sql.strip().rstrip(";") + ";"
            try:
                cur.execute(sql_clean)
                try:
                    rows = cur.fetchmany(limit_preview)
                except psycopg2.ProgrammingError:
                    # no results (e.g., DDL or COUNT only), ignore
                    rows = []
                print(f"[{total:03d}] {qid}: ✅ OK", end="")
                if rows:
                    print(f" — preview {min(len(rows), limit_preview)} row(s): {rows}")
                else:
                    print("")
                passed += 1
            except Exception as e:
                print(f"[{total:03d}] {qid}: ❌ ERROR — {e}")
                failed += 1

    print("\n=== SUMMARY ===")
    print(f"Total lines: {total}")
    print(f"Passed SQL:  {passed}")
    print(f"Failed SQL:  {failed}")
    cur.close()
    conn.close()
    return 0 if failed == 0 else 1

def main():
    ap = argparse.ArgumentParser(description="Validate gold_sql in a JSONL file against a Postgres database.")
    ap.add_argument("jsonl_path", help="Path to JSONL file (e.g., test_set.jsonl)")
    ap.add_argument("--timeout-ms", type=int, default=int(os.environ.get("PG_STMT_TIMEOUT_MS", "60000")),
                    help="statement_timeout in milliseconds (default: 60000)")
    ap.add_argument("--preview", type=int, default=int(os.environ.get("PREVIEW_ROWS", "3")),
                    help="Preview up to N rows from SELECT results (default: 3)")
    args = ap.parse_args()
    sys.exit(validate_jsonl(args.jsonl_path, args.timeout_ms, args.preview))

if __name__ == "__main__":
    main()

# Run validation
# python scripts/validate_gold_sql.py data/annotate/test_set.jsonl
# python scripts/validate_gold_sql.py data/annotate/example_set.jsonl
# python scripts/validate_gold_sql.py data/annotate/train_set.jsonl