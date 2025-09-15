from __future__ import annotations
import argparse, json, re
from pathlib import Path

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

def reason_bucket(d: dict) -> str:
    err = (d.get("error") or d.get("esm_error") or "").strip()
    esm = d.get("esm")
    if "Non-SELECT" in err:
        return "non_select"
    if "GOLD exec fail" in err or "gold failed" in err.lower():
        return "gold_fail"
    if "PRED exec fail" in err or "pred failed" in err.lower():
        return "pred_fail"
    if "timeout" in err.lower():
        return "timeout"
    if esm is False:
        return "mismatch"
    if esm is None:
        return "skipped"
    return "ok" if not err else "other_error"

def main():
    ap = argparse.ArgumentParser(description="Inspect eval details.jsonl and break down ESM issues.")
    ap.add_argument("--run_dir", default=None, help="Specific run directory (e.g., run/run_20250908_001148)")
    ap.add_argument("--runs_root", default="run", help="Root folder containing run_* dirs (default: run/)")
    ap.add_argument("--max_examples", type=int, default=5, help="Max examples per bucket to print")
    args = ap.parse_args()

    rd = Path(args.run_dir) if args.run_dir else latest_run_dir(Path(args.runs_root))
    if not rd:
        raise SystemExit(f"No run directory found. Checked --run_dir={args.run_dir!r} and --runs_root='{args.runs_root}'.")
    print(f"[inspect] run_dir = {rd}")

    details = load_details(rd)
    buckets = {k:0 for k in ["non_select","gold_fail","pred_fail","timeout","mismatch","skipped","ok","other_error"]}
    examples = {k: [] for k in buckets.keys()}

    for d in details:
        b = reason_bucket(d)
        buckets[b] += 1
        if len(examples[b]) < args.max_examples:
            examples[b].append({
                "idx": d.get("idx"),
                "em": d.get("em"),
                "esm": d.get("esm"),
                "etm": d.get("etm"),
                "error": d.get("error") or d.get("esm_error"),
                "gold": (d.get("gold_sql") or "")[:300],
                "pred": (d.get("pred_sql") or "")[:300],
            })

    total = len(details)
    print("\n=== ESM issue breakdown ===")
    for k in ["ok","mismatch","skipped","non_select","gold_fail","pred_fail","timeout","other_error"]:
        pct = (buckets[k]/total*100 if total else 0)
        print(f"{k:>12}: {buckets[k]:>3}  ({pct:5.1f}%)")

    print("\n=== Examples (up to --max_examples each) ===")
    for k in ["non_select","gold_fail","pred_fail","timeout","mismatch","skipped","other_error"]:
        if examples[k]:
            print(f"\n[{k}]")
            for ex in examples[k]:
                print(f"- idx={ex['idx']} em={ex['em']} esm={ex['esm']} etm={ex['etm']}")
                if ex["error"]:
                    print(f"  error: {ex['error']}")
                print("  GOLD:", ex["gold"])
                print("  PRED:", ex["pred"])

if __name__ == "__main__":
    main()
