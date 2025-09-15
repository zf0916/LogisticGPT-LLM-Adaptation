# evalx/em.py
from .normalization import normalize_sql_for_em
from .eval_utils import _is_unanswerable_sql, LineEval

def evaluate_em(preds, golds, details: list[LineEval]):
    n = min(len(preds), len(golds))
    for i in range(n):
        p = preds[i]
        g = golds[i]

        # Unanswerable logic
        gold_unans = _is_unanswerable_sql(g)
        pred_unans = _is_unanswerable_sql(p)
        if gold_unans and pred_unans:
            details[i].em = True
            continue
        if gold_unans ^ pred_unans:
            details[i].em = False
            continue

        # Normal EM compare (after normalization)
        ps = normalize_sql_for_em(p)
        gs = normalize_sql_for_em(g)
        if not ps or not gs:
            details[i].em = None  # skipped / incomplete
        else:
            details[i].em = (ps == gs)
