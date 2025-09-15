from __future__ import annotations
import re
from typing import List, Optional
from .eval_utils import LineEval, _is_unanswerable_sql

# Optional deep intent matcher via sqlglot (if available)
try:
    import sqlglot
    import sqlglot.optimizer as opt
    _HAS_SQLGLOT = True
except Exception:
    _HAS_SQLGLOT = False


def _skeleton(sql: str) -> str:
    """Fallback 'intent-ish' skeleton: strip literals / numbers.
    This is NOT equal to ETM paper, but provides a light proxy.
    """
    s = sql.strip()
    # remove strings and numbers
    s = re.sub(r"'[^']*'", "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "?", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _sqlglot_norm(sql: str) -> Optional[str]:
    if not _HAS_SQLGLOT:
        return None
    try:
        e = sqlglot.parse_one(sql, read="postgres")
        # basic normalizations
        e = opt.qualify.qualify(e)
        e = opt.normalize.normalizer.normalize(e)
        e = opt.prune.prune_columns(e)
        e = opt.simplify.simplify(e)
        return e.sql(dialect="postgres")
    except Exception:
        return None


def evaluate_etm(preds: List[str], golds: List[str], details: List[LineEval]):
    n = min(len(preds), len(golds))
    for i in range(n):
        p = preds[i]
        g = golds[i]

        # Unanswerable logic
        gold_unans = _is_unanswerable_sql(g)
        pred_unans = _is_unanswerable_sql(p)
        if gold_unans and pred_unans:
            details[i].etm = True
            continue
        if gold_unans ^ pred_unans:
            details[i].etm = False
            continue
        # Fallback skeleton compare
        details[i].etm = (_skeleton(p) == _skeleton(g))