"""Per-group metrics and the gaps between them.

Three ideas of fair, all reported:
- demographic parity: does it say yes to each group at the same rate
- equalised odds: does it make each kind of mistake at the same rate
- calibration: when it says 70%, is it right about 70% of the time, in every group
They can't all hold when base rates differ, and here they do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import INTERSECTION, RACE_LABELS, SEX_LABELS, intersection_label
from src.model import expected_calibration_error

# Groups smaller than this are reported but never used to draw a conclusion.
MIN_GROUP_N = 500


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def group_table(
    y: np.ndarray, p: np.ndarray, pred: np.ndarray, groups: pd.Series, attribute: str
) -> pd.DataFrame:
    """Per-group performance, error rates and calibration."""
    if attribute == INTERSECTION:
        def label_of(c):
            return intersection_label(int(c))
    else:
        table_ = RACE_LABELS if attribute == "RAC1P" else SEX_LABELS

        def label_of(c):
            return table_.get(int(c), f"code {int(c)}")

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y,
        y_pred=pred,
        sensitive_features=groups,
    )

    rows = []
    for code, metrics in frame.by_group.iterrows():
        mask = (groups == code).to_numpy()
        n = int(mask.sum())
        rows.append(
            {
                "attribute": attribute,
                "code": int(code),
                "group": label_of(code),
                "n": n,
                "base_rate": float(y[mask].mean()) if n else float("nan"),
                "selection_rate": float(metrics["selection_rate"]),
                "accuracy": float(metrics["accuracy"]),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
                "auc": _safe_auc(y[mask], p[mask]) if n else float("nan"),
                "mean_predicted": float(p[mask].mean()) if n else float("nan"),
                "ece": expected_calibration_error(y[mask], p[mask]) if n else float("nan"),
                "reportable": n >= MIN_GROUP_N,
            }
        )

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def disparity_summary(
    y: np.ndarray, pred: np.ndarray, groups: pd.Series, attribute: str, table: pd.DataFrame
) -> dict:
    """Single-number gaps, computed only over groups large enough to mean anything."""
    big = table[table["reportable"]]
    keep = set(big["code"].tolist())
    mask = groups.isin(keep).to_numpy()

    y_k, pred_k, g_k = y[mask], pred[mask], groups[mask]

    return {
        "attribute": attribute,
        "groups_reported": len(keep),
        "groups_suppressed": int((~table["reportable"]).sum()),
        "demographic_parity_difference": float(
            demographic_parity_difference(y_k, pred_k, sensitive_features=g_k)
        ),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(y_k, pred_k, sensitive_features=g_k)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_k, pred_k, sensitive_features=g_k)
        ),
        "tpr_gap": float(big["tpr"].max() - big["tpr"].min()),
        "fpr_gap": float(big["fpr"].max() - big["fpr"].min()),
        "accuracy_gap": float(big["accuracy"].max() - big["accuracy"].min()),
        "calibration_gap": float(
            (big["mean_predicted"] - big["base_rate"]).abs().max()
        ),
        "base_rate_gap": float(big["base_rate"].max() - big["base_rate"].min()),
    }


def audit(y, p, pred, A: pd.DataFrame, attributes: list[str]):
    """Run the full audit for each protected attribute. Returns tables and summaries."""
    tables, summaries = [], []
    for attr in attributes:
        groups = A[attr]
        table = group_table(np.asarray(y), np.asarray(p), np.asarray(pred), groups, attr)
        tables.append(table)
        summaries.append(disparity_summary(np.asarray(y), np.asarray(pred), groups, attr, table))
    return pd.concat(tables, ignore_index=True), pd.DataFrame(summaries)
