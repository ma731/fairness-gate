"""A second model family: is the gap in the algorithm or in the data?

Fits a logistic regression on the same splits and features as the boosted trees.
Categoricals are one-hot encoded, since a linear model would otherwise read occupation
code 500 as five times code 100. The threshold is picked on validation like everything
else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED
from src.data import CATEGORICAL, NUMERIC, Split
from src.model import pick_threshold
from src.unaware import REPORTING_FLOOR, _gaps


def build_pipeline() -> Pipeline:
    """One-hot for the codes, standardised numbers, and a linear model on top."""
    return Pipeline([
        ("prep", ColumnTransformer([
            # Unseen categories become all-zero rather than an error, which is the same
            # thing _align_categories does for the tree: a category that only appears in
            # a later year must not blow up at prediction time.
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30),
             CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])),
        ("lr", LogisticRegression(
            max_iter=400,
            C=1.0,
            solver="lbfgs",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )),
    ])


def _as_frame(split: Split) -> pd.DataFrame:
    """Categories back to plain values, because the encoder wants the raw codes."""
    X = split.X.copy()
    for column in CATEGORICAL:
        if str(X[column].dtype) == "category":
            X[column] = X[column].astype(object)
    return X


def build(splits: dict[str, Split], attribute: str = "RAC1P",
          floor: int = REPORTING_FLOOR) -> dict:
    """Fit the linear model and measure it exactly as the tree was measured."""
    model = build_pipeline()
    model.fit(_as_frame(splits["train"]), splits["train"].y)

    p_val = model.predict_proba(_as_frame(splits["val"]))[:, 1]
    threshold = pick_threshold(splits["val"].y.to_numpy(), p_val)

    test = splits["test"]
    p_test = model.predict_proba(_as_frame(test))[:, 1]
    y = test.y.to_numpy()
    pred = (p_test >= threshold).astype(int)

    from sklearn.metrics import roc_auc_score

    out = _gaps(y, pred, test.A[attribute], floor)
    out.update({
        "family": "logistic regression",
        "attribute": attribute,
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == y).mean()), 6),
        "auc": round(float(roc_auc_score(y, p_test)), 6),
        "n": len(y),
        "features": list(test.X.columns),
    })
    return out


def compare(tree: dict, linear: dict) -> dict:
    """Did changing the model family move the gap, or only the accuracy?"""
    if not linear or linear.get("tpr_gap") is None:
        return {}
    before, after = float(tree["tpr_gap"]), float(linear["tpr_gap"])
    return {
        "tpr_gap_tree": round(before, 6),
        "tpr_gap_linear": round(after, 6),
        "auc_tree": round(float(tree["auc"]), 6),
        "auc_linear": round(float(linear["auc"]), 6),
        "accuracy_tree": round(float(tree["accuracy"]), 6),
        "accuracy_linear": round(float(linear["accuracy"]), 6),
        # The line that matters: how much of the disparity is still there after
        # swapping out the algorithm entirely.
        "share_remaining": round(after / before, 6) if before else None,
    }


def rank_correlation(tree_groups: dict, linear_groups: dict) -> float | None:
    """Spearman correlation of the groups' recall ranks under the two models.

    Equal gaps could still come from failing different people. This checks it's the same
    groups at the bottom.
    """
    shared = sorted(set(tree_groups) & set(linear_groups))
    if len(shared) < 3:
        return None
    a = [tree_groups[k]["tpr"] for k in shared]
    b = [linear_groups[k]["tpr"] for k in shared]
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)
