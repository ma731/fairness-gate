"""Tests for the second model family.

Fitting a logistic regression on 576,000 rows is far too slow for CI, so what runs here
is the comparison arithmetic and the committed result. The finding itself gets a test,
because if it ever inverts then several paragraphs on the site become wrong and should
fail loudly rather than quietly mislead.
"""

from __future__ import annotations

import json

import pytest

from src import compare, report

AUDIT_PATH = report.RESULTS_DIR / "audit.json"

pytestmark = pytest.mark.skipif(
    not AUDIT_PATH.exists(), reason="no committed audit; run scripts/run_audit.py"
)


@pytest.fixture(scope="module")
def result():
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_the_audit_carries_the_linear_arm(result):
    arm = result.get("linear") or {}
    assert arm, "no linear arm; the second family did not run"
    assert arm["family"] == "logistic regression"
    assert arm["tpr_gap"] is not None


def test_both_families_saw_the_same_features(result):
    """A different feature set would make the comparison meaningless."""
    linear = set(result["linear"]["features"])
    unaware = set(result["unaware"]["features"])
    assert "RAC1P" in linear, "the linear model must match the shipped model, not the unaware one"
    assert linear != unaware


def test_the_linear_model_picked_its_own_threshold(result):
    """Reusing the tree's cut-off would compare two models at two operating points."""
    assert result["linear"]["threshold"] != result["design"]["threshold"]


def test_changing_the_algorithm_does_not_close_the_gap(result):
    """The finding. If this inverts, the method page is wrong."""
    c = result["linear"]["comparison"]
    assert c["share_remaining"] > 0.8, (
        "a different model family closed most of the gap, which would make this a fact "
        "about gradient boosting rather than about the data, and several paragraphs on "
        "the site would need rewriting"
    )


def test_the_two_families_fail_the_same_groups(result):
    """Equal gaps could still mean the models are hurting different people."""
    rho = result["linear"]["rank_correlation"]
    assert rho is not None
    assert rho > 0.7, f"the two families rank the groups differently (rho={rho})"


def test_comparison_arithmetic():
    tree = {"tpr_gap": 0.30, "auc": 0.90, "accuracy": 0.80}
    linear = {"tpr_gap": 0.27, "auc": 0.86, "accuracy": 0.78}
    c = compare.compare(tree, linear)
    assert c["share_remaining"] == pytest.approx(0.9)
    assert c["auc_linear"] == pytest.approx(0.86)


def test_comparison_is_empty_without_an_arm():
    assert compare.compare({"tpr_gap": 0.3, "auc": 0.9, "accuracy": 0.8}, {}) == {}


def test_rank_correlation_spots_a_reversal():
    a = {"1": {"tpr": 0.9}, "2": {"tpr": 0.7}, "3": {"tpr": 0.5}, "4": {"tpr": 0.3}}
    same = {"1": {"tpr": 0.8}, "2": {"tpr": 0.6}, "3": {"tpr": 0.4}, "4": {"tpr": 0.2}}
    flipped = {"1": {"tpr": 0.2}, "2": {"tpr": 0.4}, "3": {"tpr": 0.6}, "4": {"tpr": 0.8}}
    assert compare.rank_correlation(a, same) == pytest.approx(1.0)
    assert compare.rank_correlation(a, flipped) == pytest.approx(-1.0)


def test_rank_correlation_declines_to_guess_from_two_points():
    a = {"1": {"tpr": 0.9}, "2": {"tpr": 0.7}}
    assert compare.rank_correlation(a, a) is None


def test_the_pipeline_one_hot_encodes_the_codes():
    """A linear model reads occupation code 500 as five times code 100 otherwise."""
    steps = dict(compare.build_pipeline().named_steps)
    prep = steps["prep"]
    names = [name for name, _, _ in prep.transformers]
    assert "cat" in names and "num" in names
