"""Tests for the fairness-through-unawareness experiment.

The experiment exists because I got this wrong in the documentation first: I wrote that
the model never saw race or sex, when `run_audit.py` had been passing the default all
along and it did. These tests are what stops a claim like that going back in unchecked.

Training is far too slow for CI, so what runs here is the arithmetic and the committed
result, not the fit.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import report, unaware

AUDIT_PATH = report.RESULTS_DIR / "audit.json"

pytestmark = pytest.mark.skipif(
    not AUDIT_PATH.exists(), reason="no committed audit; run scripts/run_audit.py"
)


@pytest.fixture(scope="module")
def result():
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_the_audit_carries_both_arms(result):
    arm = result.get("unaware") or {}
    assert arm, "the audit has no unaware arm; the experiment did not run"
    assert arm["tpr_gap"] is not None
    assert arm["comparison"]["tpr_gap_aware"] > 0


def test_the_unaware_model_really_lost_the_columns(result):
    """If the drop silently failed, every number below would be meaningless."""
    features = result["unaware"]["features"]
    assert "RAC1P" not in features
    assert "SEX" not in features


def test_the_shipped_model_really_kept_them(result):
    """The other half of the claim, and the half I originally got backwards."""
    design = result["design"]
    assert design.get("drop_protected") is not True


def test_most_of_the_gap_survives_removing_the_attributes(result):
    """The finding itself. If this ever inverts, the story on the site is wrong."""
    c = result["unaware"]["comparison"]
    assert c["share_remaining"] > 0.5, (
        "deleting race and sex closed more than half the gap, which would make "
        "unawareness a real mitigation and every page saying otherwise wrong"
    )


def test_comparison_arithmetic():
    aware = {"tpr_gap": 0.30, "accuracy": 0.80}
    arm = {"tpr_gap": 0.24, "accuracy": 0.79}
    c = unaware.compare(aware, arm)
    assert c["closed"] == pytest.approx(0.06)
    assert c["share_remaining"] == pytest.approx(0.8)
    assert c["accuracy_cost"] == pytest.approx(0.01)


def test_comparison_is_empty_when_there_is_nothing_to_compare():
    assert unaware.compare({"tpr_gap": 0.3, "accuracy": 0.8}, {}) == {}


def test_small_groups_are_excluded_from_the_comparison():
    """The floor has to apply here too, or the experiment can be steered by noise."""
    y = pd.Series([1, 0] * 60).to_numpy()
    pred = pd.Series([1, 0] * 60).to_numpy()
    groups = pd.Series([1] * 100 + [2] * 20)
    out = unaware._gaps(y, pred, groups, floor=50)
    assert "1" in out["groups"]
    assert "2" not in out["groups"], "a 20 person group was allowed into the comparison"
