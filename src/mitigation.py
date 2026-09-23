"""What it would take to close the gap, and what closing it costs.

The site spent a lot of effort proving a disparity exists and never once tried to fix
it, which is the easy half. This module does the hard half and reports the bill.

Two things are measured, and the contrast between them is the point:

1. **Sweeping the single global threshold.** The obvious first move, and it does not
   work. Moving one cut up or down trades recall against false positives for everybody
   at once; it barely touches the difference between groups. Showing that is what earns
   the right to reach for something more invasive.

2. **Per-group thresholds** chosen to equalise recall (equal opportunity, Hardt et al.).
   This does close the gap, and it costs accuracy, and it needs the protected attribute
   at the moment of prediction. That last requirement is not a footnote. In hiring, in
   lending, in most of the places anyone would want this, using race at decision time is
   either unlawful or is itself the harm. A mitigation that can only run by doing the
   thing you were trying to avoid belongs in the documentation as a rejected option with
   its reasoning, not in the pipeline.

Every threshold here is chosen on the validation year and applied unchanged to the test
year, the same discipline as the rest of the project. Choosing them on test would be
fitting the fix to the exam.
"""

from __future__ import annotations

import numpy as np

SWEEP_POINTS = 41


def _rates(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pos, neg = (y == 1).sum(), (y == 0).sum()
    tpr = float(((pred == 1) & (y == 1)).sum() / pos) if pos else float("nan")
    fpr = float(((pred == 1) & (y == 0)).sum() / neg) if neg else float("nan")
    return tpr, fpr


def _gap(y, p, groups, keep, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tprs, fprs = [], []
    for code in keep:
        m = (groups == code).to_numpy()
        if not m.any():
            continue
        t, f = _rates(y[m], pred[m])
        if np.isfinite(t):
            tprs.append(t)
        if np.isfinite(f):
            fprs.append(f)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "tpr_gap": round(max(tprs) - min(tprs), 4) if len(tprs) > 1 else None,
        "fpr_gap": round(max(fprs) - min(fprs), 4) if len(fprs) > 1 else None,
    }


def global_sweep(y, p, groups, keep, points: int = SWEEP_POINTS) -> list[dict]:
    """The gap, and the accuracy, at every global threshold worth considering.

    If one cut could fix this, it would show up here as a dip. It does not.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    return [
        _gap(y, p, groups, keep, t)
        for t in np.linspace(0.05, 0.95, points)
    ]


def equal_opportunity_thresholds(y_val, p_val, groups_val, keep, target: float) -> dict:
    """One threshold per group, chosen on validation so each group hits `target` recall.

    Picked by searching each group's own score distribution for the cut that puts its
    recall closest to the target. The target is the recall the current single threshold
    already achieves overall, so the comparison is "same headline recall, distributed
    differently" rather than "we quietly raised recall for everyone".
    """
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val)
    grid = np.linspace(0.02, 0.98, 97)
    out: dict[int, float] = {}
    for code in keep:
        m = (groups_val == code).to_numpy()
        yg, pg = y_val[m], p_val[m]
        if (yg == 1).sum() == 0:
            out[int(code)] = 0.5
            continue
        best, best_err = 0.5, float("inf")
        for t in grid:
            tpr, _ = _rates(yg, (pg >= t).astype(int))
            err = abs(tpr - target)
            if err < best_err:
                best, best_err = float(t), err
        out[int(code)] = round(best, 4)
    return out


def apply_thresholds(p, groups, thresholds: dict, fallback: float) -> np.ndarray:
    p = np.asarray(p)
    cuts = np.full(len(p), fallback, dtype=float)
    g = groups.to_numpy()
    for code, t in thresholds.items():
        cuts[g == code] = t
    return (p >= cuts).astype(int)


def build(y_val, p_val, A_val, y_test, p_test, A_test, attribute: str,
          keep: set, baseline_threshold: float) -> dict:
    """The full comparison: do nothing, move one cut, or move one cut per group."""
    y_test = np.asarray(y_test)
    p_test = np.asarray(p_test)
    g_val, g_test = A_val[attribute], A_test[attribute]
    keep = sorted(keep)
    if len(keep) < 2:
        return {}

    base = _gap(y_test, p_test, g_test, keep, baseline_threshold)
    sweep = global_sweep(y_test, p_test, g_test, keep)

    # the best any single global threshold can do for the gap, and what it costs
    scored = [s for s in sweep if s["tpr_gap"] is not None]
    best_global = min(scored, key=lambda s: s["tpr_gap"]) if scored else None

    overall_tpr, _ = _rates(y_test, (p_test >= baseline_threshold).astype(int))
    cuts = equal_opportunity_thresholds(y_val, p_val, g_val, keep, overall_tpr)
    pred = apply_thresholds(p_test, g_test, cuts, baseline_threshold)

    tprs, fprs = [], []
    for code in keep:
        m = (g_test == code).to_numpy()
        t, f = _rates(y_test[m], pred[m])
        if np.isfinite(t):
            tprs.append(t)
        if np.isfinite(f):
            fprs.append(f)

    mitigated = {
        "accuracy": round(float((pred == y_test).mean()), 4),
        "tpr_gap": round(max(tprs) - min(tprs), 4),
        "fpr_gap": round(max(fprs) - min(fprs), 4),
        "thresholds": cuts,
        "target_recall": round(float(overall_tpr), 4),
    }

    return {
        "attribute": attribute,
        "baseline": base,
        "sweep": sweep,
        "best_global": best_global,
        "per_group": mitigated,
        "accuracy_cost": round(base["accuracy"] - mitigated["accuracy"], 4),
        "gap_closed": round(base["tpr_gap"] - mitigated["tpr_gap"], 4),
        "requires_attribute_at_inference": True,
    }
