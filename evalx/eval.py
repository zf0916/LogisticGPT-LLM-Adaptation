# evalx/main.py
from __future__ import annotations
import argparse
import re
from pathlib import Path

from .canonicalization import write_canonicalized_files
from .eval_utils import load_lines, LineEval, write_jsonl, write_json, summarize_detail, to_records
from .em import evaluate_em
from .esm import evaluate_esm


def parse_args():
    ap = argparse.ArgumentParser(description="Simple, modular evaluation pipeline (EM, ESM)")
    ap.add_argument("--run_dir", default=None, help="Path to a single run directory (with preds/gold files)")
    ap.add_argument("--runs_root", default="run", help="Root containing run_* dirs (used if --run_dir not set)")
    ap.add_argument("--dsn", default="postgresql://postgres:postgres@localhost:5433/t2sql_db", help="Postgres DSN for ESM (optional)")
    ap.add_argument("--exec_timeout", type=int, default=30, help="ESM execution timeout (seconds)")
    ap.add_argument("--no_em", action="store_true", help="Disable EM")
    ap.add_argument("--no_esm", action="store_true", help="Disable ESM")
    return ap.parse_args()


def _latest_run_dir(runs_root: Path) -> Path | None:
    """Pick latest run_* by timestamp in name; else by mtime."""
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    pat = re.compile(r"^run_(\d{8})_(\d{6})$")  # run_YYYYMMDD_HHMMSS
    named = [p for p in candidates if pat.match(p.name)]
    if named:
        return sorted(named, key=lambda p: p.name)[-1]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    args = parse_args()
    rd = Path(args.run_dir) if args.run_dir else _latest_run_dir(Path(args.runs_root))
    if rd is None:
        raise SystemExit(f"No run directory found. Checked --run_dir={args.run_dir!r} and --runs_root='{args.runs_root}'.")
    print(f"[evalx] Using run_dir: {rd}")

    out_dir = rd / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Canonicalize
    _ = write_canonicalized_files(rd)

    # 2) Read canonicalized lines for evaluation (fallback to originals if canon missing)
    canon_preds = out_dir / "canon" / "preds.txt"
    canon_golds = out_dir / "canon" / "gold.txt"
    preds = load_lines(canon_preds) if canon_preds.exists() else load_lines(rd / "preds.txt")
    golds = load_lines(canon_golds) if canon_golds.exists() else load_lines(rd / "gold.txt")

    n = min(len(preds), len(golds))
    details = [LineEval(idx=i, pred_sql=preds[i], gold_sql=golds[i]) for i in range(n)]

    # 3) Metrics
    if not args.no_em:
        evaluate_em(preds, golds, details)
    if not args.no_esm:
        evaluate_esm(args.dsn, preds, golds, details, exec_timeout=args.exec_timeout)

    # 4) Write outputs
    write_jsonl(out_dir / "details.jsonl", to_records(details))
    summary = summarize_detail(details)
    write_json(out_dir / "eval_summary.json", summary)

    print("=== EVAL SUMMARY ===")
    total  = summary.get("total", 0)
    skips  = summary.get("skips", {})
    rates  = summary.get("rates", {})
    counts = summary.get("counts", {})

    print(f"total: {total}")
    print(f"empty_preds: {skips.get('empty_pred', 0)} ; unanswerable: {skips.get('unanswerable', 0)}")

    def _line(metric: str) -> str:
        c = counts.get(metric, {})
        r = rates.get(metric, {})
        pass_cnt = c.get("pass", 0)
        attempted = r.get("attempted", 0)
        pct_total = r.get("pass_pct_total", 0.0)
        pct_attempted = r.get("pass_pct_attempted", 0.0)
        # Example: "EM - 4/30 13.3% (of attempted: 4/30 13.3%)"
        return f"{metric.upper()} - {pass_cnt}/{total} {pct_total}% (of attempted: {pass_cnt}/{attempted} {pct_attempted}%)"

    print(_line("em"))
    print(_line("esm"))
    print()  # blank line

    b = summary.get("esm_breakdown")
    if b:
        print("=== ESM breakdown ===")
        for k in ["ok","mismatch","skipped","non_select","gold_fail","pred_fail","timeout","other_error"]:
            print(f"{k:>12}: {b.get(k, 0)}")


if __name__ == "__main__":
    main()
