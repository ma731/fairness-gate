"""Load ACS income data and build the train / validation / test splits.

The task is the standard folktables ACSIncome problem: predict whether a person's
income exceeds $50,000, from ten census variables. Real survey data, real ground
truth, real distribution shift between years and between states.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

warnings.filterwarnings("ignore")

from folktables import ACSDataSource, ACSIncome

from src.config import (
    CACHE_DIR,
    SHIFT_STATES,
    TEST_YEAR,
    TRAIN_STATES,
    TRAIN_YEAR,
    VAL_YEAR,
)

# Codes that are categorical despite being stored as integers. Trees handle these
# natively; a linear model would read them as ordinal and quietly learn nonsense.
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
NUMERIC = ["AGEP", "WKHP"]

# Attributes we audit across. Kept separate from "features" on purpose: an attribute
# can be protected whether or not the model is allowed to see it.
PROTECTED = ["RAC1P", "SEX"]


@dataclass(frozen=True)
class Split:
    """One slice of data, with features, labels and protected attributes kept apart."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    A: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)

    def describe(self) -> str:
        return f"{self.name:<18} n={len(self):>8,}  positive rate={self.y.mean():.3f}"


def _load(year: str, states: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    source = ACSDataSource(
        survey_year=year, horizon="1-Year", survey="person", root_dir=str(CACHE_DIR)
    )
    raw = source.get_data(states=states, download=False)
    X, y, _ = ACSIncome.df_to_pandas(raw)

    y = y.iloc[:, 0].astype(int)
    A = X[PROTECTED].copy()

    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    for col in NUMERIC:
        X[col] = X[col].astype(float)

    return X.reset_index(drop=True), y.reset_index(drop=True), A.reset_index(drop=True)


def _align_categories(splits: list[Split]) -> list[Split]:
    """Force one shared category set per column, taken from all splits.

    Without this, a category that appears in 2018 but not 2015 becomes a silent NaN
    at prediction time and the model looks better than it is.
    """
    out = []
    for col in CATEGORICAL:
        levels = sorted(
            set().union(*[set(s.X[col].cat.categories.tolist()) for s in splits])
        )
        for s in splits:
            s.X[col] = s.X[col].cat.set_categories(levels)
    for s in splits:
        out.append(s)
    return out


def load_splits(drop_protected: bool = False) -> dict[str, Split]:
    """Build the four splits.

    train        2015, five large states
    val          2016, same states. Used for threshold and calibration choices.
    test         2018, same states. Temporal shift only.
    shift        2018, four different states. Temporal plus geographic shift.

    drop_protected removes race and sex from the features while keeping them for
    auditing. That is the "fairness through unawareness" variant.
    """
    specs = [
        ("train", TRAIN_YEAR, TRAIN_STATES),
        ("val", VAL_YEAR, TRAIN_STATES),
        ("test", TEST_YEAR, TRAIN_STATES),
        ("shift", TEST_YEAR, SHIFT_STATES),
    ]

    splits = []
    for name, year, states in specs:
        X, y, A = _load(year, states)
        splits.append(Split(name=name, X=X, y=y, A=A))

    splits = _align_categories(splits)

    if drop_protected:
        splits = [
            Split(s.name, s.X.drop(columns=PROTECTED), s.y, s.A) for s in splits
        ]

    return {s.name: s for s in splits}


def feature_columns(drop_protected: bool = False) -> list[str]:
    cols = NUMERIC + CATEGORICAL
    if drop_protected:
        cols = [c for c in cols if c not in PROTECTED]
    return cols
