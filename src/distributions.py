"""Extra series the charts need, computed once during the audit and committed.

The page is rendered from results/audit.json alone, so anything a chart wants has to be
exported here first. Everything in this module is a summary: binned counts and rates,
never a row about a person.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Enough bins to show a shape, few enough that every bin holds real weight.
SCORE_BINS = 20
CALIB_BINS = 10
ROC_POINTS = 40


def _safe(v) -> float:
    return float(v) if np.isfinite(v) else 0.0


def score_histogram(y: np.ndarray, p: np.ndarray, groups: pd.Series,
                    keep: set, bins: int = SCORE_BINS) -> list[dict]:
    """Predicted-probability distribution per group, split by the true outcome.

    This is what a ridgeline plot draws. Two overlapping shapes per group: the people who
    qualify and the people who do not. Where they overlap is where the model cannot tell
    them apart, and that overlap differing by group is the whole story.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for code in sorted(keep):
        m = (groups == code).to_numpy()
        if not m.any():
            continue
        pg, yg = p[m], y[m]
        pos, _ = np.histogram(pg[yg == 1], bins=edges)
        neg, _ = np.histogram(pg[yg == 0], bins=edges)
        out.append({
            "code": int(code),
            "positive": [int(v) for v in pos],
            "negative": [int(v) for v in neg],
        })
    return out


def calibration_curve(y: np.ndarray, p: np.ndarray, groups: pd.Series,
                      keep: set, bins: int = CALIB_BINS) -> list[dict]:
    """Reliability: mean predicted against observed, per equal-width bin, per group.

    A perfectly calibrated model traces the diagonal. Bins holding fewer than 50 people
    are dropped rather than plotted as noise.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for code in sorted(keep):
        m = (groups == code).to_numpy()
        pg, yg = p[m], y[m]
        pts = []
        idx = np.digitize(pg, edges[1:-1], right=True)
        for b in range(bins):
            sel = idx == b
            n = int(sel.sum())
            if n < 50:
                continue
            pts.append({
                "n": n,
                "predicted": round(_safe(pg[sel].mean()), 4),
                "observed": round(_safe(yg[sel].mean()), 4),
            })
        if pts:
            out.append({"code": int(code), "points": pts})
    return out


def roc_curve(y: np.ndarray, p: np.ndarray, groups: pd.Series,
              keep: set, points: int = ROC_POINTS) -> list[dict]:
    """ROC per group, sampled at a fixed set of thresholds.

    Fixed thresholds rather than every unique score: the curves stay comparable across
    groups and the committed file stays small.
    """
    cuts = np.linspace(0.02, 0.98, points)
    out = []
    for code in sorted(keep):
        m = (groups == code).to_numpy()
        pg, yg = p[m], y[m]
        pos, neg = (yg == 1).sum(), (yg == 0).sum()
        if pos == 0 or neg == 0:
            continue
        pts = []
        for c in cuts:
            pred = pg >= c
            tpr = _safe(((pred == 1) & (yg == 1)).sum() / pos)
            fpr = _safe(((pred == 1) & (yg == 0)).sum() / neg)
            pts.append({"t": round(float(c), 3), "tpr": round(tpr, 4),
                        "fpr": round(fpr, 4)})
        out.append({"code": int(code), "points": pts})
    return out


def confusion_by_group(y: np.ndarray, pred: np.ndarray, groups: pd.Series,
                       keep: set) -> list[dict]:
    """Raw confusion counts per group. The Sankey and the heatmap both read this.

    Counts, not rates: the flow diagram has to conserve people, and a rate cannot.
    """
    out = []
    for code in sorted(keep):
        m = (groups == code).to_numpy()
        yg, pg = y[m], pred[m]
        out.append({
            "code": int(code),
            "tp": int(((pg == 1) & (yg == 1)).sum()),
            "fn": int(((pg == 0) & (yg == 1)).sum()),
            "fp": int(((pg == 1) & (yg == 0)).sum()),
            "tn": int(((pg == 0) & (yg == 0)).sum()),
        })
    return out


def build(y, p, pred, A: pd.DataFrame, attribute: str, reportable: set) -> dict:
    """Everything the distribution charts need, for one protected attribute."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = np.asarray(pred)
    g = A[attribute]
    return {
        "attribute": attribute,
        "bins": SCORE_BINS,
        "histogram": score_histogram(y, p, g, reportable),
        "calibration": calibration_curve(y, p, g, reportable),
        "roc": roc_curve(y, p, g, reportable),
        "confusion": confusion_by_group(y, pred, g, reportable),
    }
