"""Train the classifier, calibrate it, and score it.

The threshold and the calibration are both fitted on the validation year, never the test
year. Calibrated means a score of 0.7 is right about 70% of the time. That matters
because ranking people well (AUC) doesn't guarantee the probabilities are usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import RANDOM_SEED
from src.data import Split


@dataclass
class Scores:
    """Threshold-free and threshold-dependent metrics for one split."""

    split: str
    n: int
    auc: float
    average_precision: float
    brier: float
    ece: float
    accuracy: float
    threshold: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {
            "split": self.split,
            "n": self.n,
            "auc": round(self.auc, 4),
            "average_precision": round(self.average_precision, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "accuracy": round(self.accuracy, 4),
            "threshold": round(self.threshold, 4),
        }
        d.update(self.extra)
        return d


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Standard ECE: average gap between confidence and accuracy, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The threshold that maximises Youden's J (recall minus false positive rate) on validation.

    Not 0.5, because the base rate is nowhere near 0.5. J weighs both errors equally, which
    is easier to defend when one cut applies to every group.
    """
    grid = np.linspace(0.05, 0.95, 181)
    best, best_j = 0.5, -np.inf
    for t in grid:
        pred = p >= t
        tp = float(((pred == 1) & (y == 1)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1.0)
        fpr = fp / max(fp + tn, 1.0)
        j = tpr - fpr
        if j > best_j:
            best, best_j = float(t), j
    return best


def train(train_split: Split, val_split: Split, calibrate: bool = True):
    """Fit LightGBM with early stopping, then calibrate on the validation year."""
    base = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    base.fit(
        train_split.X,
        train_split.y,
        eval_set=[(val_split.X, val_split.y)],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    if not calibrate:
        return base

    # FrozenEstimator keeps the fitted model fixed, so the calibrator only ever sees the
    # validation year and never refits on it. (sklearn removed cv="prefit" in 1.8.)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(val_split.X, val_split.y)
    return calibrated


def predict_proba(model, split: Split) -> np.ndarray:
    return model.predict_proba(split.X)[:, 1]


def score(split: Split, p: np.ndarray, threshold: float) -> Scores:
    y = split.y.to_numpy()
    return Scores(
        split=split.name,
        n=len(y),
        auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        accuracy=float(accuracy_score(y, (p >= threshold).astype(int))),
        threshold=threshold,
    )


def baseline_rate(split: Split) -> float:
    """Accuracy of always predicting the majority class. The number to beat."""
    return float(max(split.y.mean(), 1 - split.y.mean()))


def scores_frame(scores: list[Scores]) -> pd.DataFrame:
    return pd.DataFrame([s.as_row() for s in scores])
