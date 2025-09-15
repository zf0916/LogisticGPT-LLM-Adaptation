from __future__ import annotations
from decimal import Decimal
import re

# --- SQL string normalization for EM (post‑canonicalization) ---
_ws = re.compile(r"\s+")
_space_before = re.compile(r"\s+([,\)])")
_space_after = re.compile(r"([\(])\s+")
_semicolon = re.compile(r";\s*$")
_comments_line = re.compile(r"--.*?$", re.MULTILINE)
_comments_block = re.compile(r"/\*.*?\*/", re.DOTALL)


def normalize_sql_for_em(sql: str | None) -> str:
    if not sql:
        return ""
    s = sql
    s = _comments_line.sub("", s)
    s = _comments_block.sub("", s)
    s = _semicolon.sub("", s)
    s = _ws.sub(" ", s)
    s = _space_before.sub(r"\1", s)
    s = _space_after.sub(r"\1", s)
    return s.strip()

# --- Result normalization for ESM (unordered set compare) ---

def _norm_scalar(v):
    # ints: leave alone
    if isinstance(v, int):
        return v
    # floats: round conservatively
    if isinstance(v, float):
        return round(v, 10)
    # NEW: decimals -> float then round
    if isinstance(v, Decimal):
        try:
            return round(float(v), 10)
        except Exception:
            return float(v)
    # strings: trim
    if isinstance(v, str):
        return v.strip()
    return v

def normalize_rows(rows: list[tuple]) -> list[tuple]:
    out = []
    for r in rows:
        out.append(tuple(_norm_scalar(x) for x in r))
    out.sort()  # unordered comparison
    return out

def normalize_sql_for_em_strict(sql: str | None) -> str:
    if not sql:
        return ""
    s = sql
    m = _prelude.search(s)
    if m:
        s = s[m.start():]
    return normalize_sql_for_em(s)
