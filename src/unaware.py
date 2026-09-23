"""The same model trained without race and sex: does removing them help?

This is fairness through unawareness, the fix most people suggest first. The second
model gets its own threshold on validation and is scored on the same test year. Groups
under the reporting floor are left out here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_splits
from src.model import pick_threshold, predict_proba, train

REPORTING_FLOOR = 500


def _gaps(y: np.ndarray, pred: np.ndarray, groups: pd.Series, floor: int) -> dict:
    """Recall and false positive rate per group, and the spread between them.

    Groups under the reporting floor are left out, same as everywhere else.
    """
    counts = groups.value_counts()
    tprs, fprs, per_group = [], [], {}
    for code in sorted(counts.index):
        if counts[code] < floor:
            continue
        m = (groups == code).to_numpy()
        yy, pp = y[m], pred[m]
        pos, neg = int((yy == 1).sum()), int((yy == 0).sum())
        if pos == 0 or neg == 0:
            continue
        tpr = float(((pp == 1) & (yy == 1)).sum() / pos)
        fpr = float(((pp == 1) & (yy == 0)).sum() / neg)
        tprs.append(tpr)
        fprs.append(fpr)
        per_group[str(int(code))] = {
            "n": int(counts[code]), "tpr": round(tpr, 6), "fpr": round(fpr, 6)
        }
    return {
        "tpr_gap": round(max(tprs) - min(tprs), 6) if len(tprs) > 1 else None,
        "fpr_gap": round(max(fprs) - min(fprs), 6) if len(fprs) > 1 else None,
        "groups": per_group,
    }


def build(attribute: str = "RAC1P", floor: int = REPORTING_FLOOR) -> dict:
    """Train without the protected columns and measure what changed.

    Returns the unaware arm only. The aware arm is already the main audit, and
    recomputing it here would risk the two disagreeing over something trivial.
    """
    splits = load_splits(drop_protected=True)
    model = train(splits["train"], splits["val"])

    p_val = predict_proba(model, splits["val"])
    threshold = pick_threshold(splits["val"].y.to_numpy(), p_val)

    test = splits["test"]
    p_test = predict_proba(model, test)
    y = test.y.to_numpy()
    pred = (p_test >= threshold).astype(int)

    out = _gaps(y, pred, test.A[attribute], floor)
    out.update({
        "attribute": attribute,
        "features": list(test.X.columns),
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == y).mean()), 6),
        "n": len(y),
    })
    return out


def compare(aware: dict, unaware: dict) -> dict:
    """The line that matters: how much of the gap survives removing the attributes."""
    if not unaware or unaware.get("tpr_gap") is None:
        return {}
    before, after = float(aware["tpr_gap"]), float(unaware["tpr_gap"])
    return {
        "tpr_gap_aware": round(before, 6),
        "tpr_gap_unaware": round(after, 6),
        "closed": round(before - after, 6),
        # the headline: the share of the disparity that deleting the columns does
        # nothing about, because the remaining features carry it anyway
        "share_remaining": round(after / before, 6) if before else None,
        "accuracy_cost": round(float(aware["accuracy"]) - float(unaware["accuracy"]), 6),
    }
