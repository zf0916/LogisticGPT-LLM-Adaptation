from __future__ import annotations
import re
from pathlib import Path

_fence_open  = re.compile(r"^\s*```(?:sql|postgresql)?\s*", re.IGNORECASE)
_fence_close = re.compile(r"\s*```\s*$", re.IGNORECASE)
_chat_open   = re.compile(r"^\s*<\s*/?s>\s*", re.IGNORECASE)          # <s> or </s>
_chat_close  = re.compile(r"\s*</\s*s>\s*$", re.IGNORECASE)
_im_start    = re.compile(r"^\s*<\|im_start\|>\s*", re.IGNORECASE)
_im_end      = re.compile(r"\s*<\|im_end\|>\s*$", re.IGNORECASE)

SQL_KEYWORDS = [
    # core
    "select","from","where","group","by","having","order","limit","offset",
    "join","inner","left","right","full","on","union","all","distinct",
    "case","when","then","else","end","and","or","not","as",
    # functions often seen
    "count","avg","sum","min","max","date_trunc"
]

_kw_re = re.compile(r"\\b(" + "|".join(map(re.escape, SQL_KEYWORDS)) + r")\\b", re.IGNORECASE)

# --- domain-friendly, safe canonical rules (pure rewrites; no schema assumptions) ---
_rules: list[tuple[re.Pattern, str]] = [
    # 1) Remove trailing semicolons + surrounding whitespace
    (re.compile(r";\\s*$"), ""),
    # 2) Normalize COUNT(1) -> COUNT(*) (but leave COUNT(DISTINCT ...) intact)
    (re.compile(r"(?i)count\\s*\\(\\s*1\\s*\\)"), "COUNT(*)"),
    # 3) Ensure AVG(...)::float (adds cast if missing)
    (re.compile(r"(?i)AVG\\s*\\((.*?)\\)\\s*(?!::\\s*float)"), r"AVG(\\1)::float"),
    # 4) Prefer / 3600.0 (avoid integer division downstream)
    (re.compile(r"/\\s*3600(?!\\.0)(?=\\D|$)"), "/ 3600.0"),
    # 5) Collapse excessive whitespace
    (re.compile(r"\\s+"), " "),
]

# For keywords, we uppercase only after other rewrites to keep things stable

def _uppercase_keywords(sql: str) -> str:
    def _up(m: re.Match) -> str:
        return m.group(1).upper()
    return _kw_re.sub(_up, sql)


def canonicalize_sql(sql: str | None) -> str:
    """Lightweight, conservative canonicalization.
    Safe: does not attempt aliasing or schema-aware rewrites.
    """
    if not sql:
        return ""
    # Strip chat/markdown wrappers that some models emit
    s = sql.strip()
    s = _fence_open.sub("", s)
    s = _fence_close.sub("", s)
    s = _chat_open.sub("", s)
    s = _chat_close.sub("", s)
    s = _im_start.sub("", s)
    s = _im_end.sub("", s)
    s = s.strip("` \n\t\r")

    for pat, repl in _rules:
        s = pat.sub(repl, s)

    # Fix missing space before LIMIT after ASC/DESC (e.g., 'ASCLIMIT' -> 'ASC LIMIT')
    s = re.sub(r"(?i)(asc)(limit)", r"\1 LIMIT", s)
    s = re.sub(r"(?i)(desc)(limit)", r"\1 LIMIT", s)
    s = re.sub(r"to_timestamp\\(\\s*([a-z_][\\w\\.]*)\\s*,\\s*'UTC'\\s*\\)", r"\\1", s, flags=re.IGNORECASE)
    s = re.sub(r"to_timestamp\(\s*([a-z_][\w\.]*)\s*,\s*'UTC'\s*\)", r"\1", s, flags=re.IGNORECASE)
    
    s = _uppercase_keywords(s)
    return s.strip()

def write_canonicalized_files(run_dir: Path) -> dict:
    """Read generation outputs, write canonicalized copies into eval/canon/.
    Returns a small dict with written paths.
    """
    rd = Path(run_dir)
    out = rd / "eval" / "canon"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "preds": (rd / "preds.txt", out / "preds.txt"),
        "golds": (rd / "gold.txt", out / "gold.txt"),
        "etm_preds": (rd / "etm_preds.txt", out / "etm_preds.txt"),
        "etm_golds": (rd / "etm_gold.txt", out / "etm_gold.txt"),
    }

    written = {}

    for key, (src, dst) in paths.items():
        if not src.exists():
            continue
        lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
        out_lines = []
        if key.startswith("etm_"):
            # format: sql \t db_id
            for line in lines:
                if "\t" in line:
                    sql, dbid = line.split("\t", 1)
                else:
                    sql, dbid = line, ""
                out_lines.append(f"{canonicalize_sql(sql)}\t{dbid}")
        else:
            out_lines = [canonicalize_sql(x) for x in lines]
        dst.write_text("\n".join(out_lines), encoding="utf-8")
        written[key] = str(dst)

    return written