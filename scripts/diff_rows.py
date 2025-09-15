from __future__ import annotations
import argparse, json, re
from pathlib import Path

# Import the project's exec_query
try:
    from evalx.eval_utils import exec_query
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from evalx.eval_utils import exec_query  # type: ignore

def latest_run_dir(root: Path) -> Path | None:
    cands = [p for p in root.iterdir() if p.is_dir()]
    if not cands:
        return None
    named = [p for p in cands if re.match(r"^run_\d{8}_\d{6}$", p.name)]
    return (sorted(named, key=lambda p: p.name)[-1] if named else
            max(cands, key=lambda p: p.stat().st_mtime))

def load_details(run_dir: Path):
    jl = run_dir / "eval" / "details.jsonl"
    if not jl.exists():
        raise SystemExit(f"Not found: {jl}")
    return [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]

def main():
    ap = argparse.ArgumentParser(description="Show GOLD vs PRED result rows for a given detail idx.")
    ap.add_argument("idx", type=int, help="detail index to inspect (0-based)")
    ap.add_argument("--runs_root", default="run", help="Root containing run_* (default: run/)")
    ap.add_argument("--run_dir", default=None, help="Specific run directory path")
    ap.add_argument("--dsn", default="postgresql://postgres:postgres@localhost:5433/t2sql_db", help="Postgres DSN")
    ap.add_argument("--limit", type=int, default=50, help="Show at most this many rows (per side)")
    args = ap.parse_args()

    rd = Path(args.run_dir) if args.run_dir else latest_run_dir(Path(args.runs_root))
    if not rd:
        raise SystemExit("Run dir not found.")
    details = load_details(rd)
    if not (0 <= args.idx < len(details)):
        raise SystemExit(f"idx {args.idx} out of range (0..{len(details)-1})")

    d = details[args.idx]
    gold = (d.get("gold_sql") or "").strip()
    pred = (d.get("pred_sql") or "").strip()
    print(f"[diff_rows] run_dir={rd} idx={args.idx}")
    print("GOLD SQL:", gold)
    print("PRED SQL:", pred)

    ok_g, rows_g, err_g = exec_query(args.dsn, gold, limit=args.limit, timeout_s=30)
    ok_p, rows_p, err_p = exec_query(args.dsn, pred, limit=args.limit, timeout_s=30)

    print("\n--- GOLD ---")
    if ok_g:
        for r in rows_g[:args.limit]:
            print(r)
        print(f"[{len(rows_g)} rows]")
    else:
        print("ERROR:", err_g)

    print("\n--- PRED ---")
    if ok_p:
        for r in rows_p[:args.limit]:
            print(r)
        print(f"[{len(rows_p)} rows]")
    else:
        print("ERROR:", err_p)

if __name__ == "__main__":
    main()
