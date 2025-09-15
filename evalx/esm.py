# evalx/esm.py
from __future__ import annotations
from typing import List, Optional
from .eval_utils import LineEval, exec_query, rows_equal_unordered, safe_select_only, _is_unanswerable_sql

def evaluate_esm(dsn: Optional[str], preds: List[str], golds: List[str], details: List[LineEval], exec_timeout: int = 30):
    if not dsn:
        return
    n = min(len(preds), len(golds))
    for i in range(n):
        p = (preds[i] or "").strip()
        g = (golds[i] or "").strip()

        # Unanswerable logic
        gold_unans = _is_unanswerable_sql(g)
        pred_unans = _is_unanswerable_sql(p)
        if gold_unans and pred_unans:
            details[i].esm = True
            continue
        if gold_unans ^ pred_unans:
            details[i].esm = False
            continue

        if not p or not g:
            details[i].esm = None
            continue

        if not (safe_select_only(p) and safe_select_only(g)):
            details[i].esm = False
            details[i].error = (details[i].error or "") + " | Non-SELECT SQL in ESM"
            continue

        ok_g, rows_g, err_g = exec_query(dsn, g, timeout_s=exec_timeout)
        if not ok_g:
            details[i].esm = None
            details[i].error = (details[i].error or "") + f" | GOLD exec fail: {err_g}"
            continue

        ok_p, rows_p, err_p = exec_query(dsn, p, timeout_s=exec_timeout)
        if not ok_p:
            details[i].esm = False
            details[i].error = (details[i].error or "") + f" | PRED exec fail: {err_p}"
            continue

        details[i].esm = rows_equal_unordered(rows_p, rows_g)
