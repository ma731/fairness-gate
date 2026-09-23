"""Tests for the gate itself.

The gate is the product here, so its failure modes matter more than the model's. The one
that would be fatal is a gate that quietly does not run: a typo in policy.yaml that
disables a fairness check while the build stays green looks exactly like coverage. Those
tests come first.
"""

from __future__ import annotations

import pytest
import yaml

from src import policy as pol
from src.policy import FAIL, PASS, WARN, PolicyError

POLICY_PATH = pol.POLICY_PATH


def _summary(attribute="RAC1P", **overrides):
    base = {
        "attribute": attribute,
        "groups_reported": 6,
        "groups_suppressed": 1,
        "demographic_parity_difference": 0.20,
        "demographic_parity_ratio": 0.5,
        "equalized_odds_difference": 0.10,
        "tpr_gap": 0.10,
        "fpr_gap": 0.05,
        "accuracy_gap": 0.04,
        "calibration_gap": 0.02,
        "base_rate_gap": 0.27,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# A gate that does not run
# --------------------------------------------------------------------------- #
def test_unknown_metric_is_fatal_not_skipped():
    """A typo in policy.yaml must break, not silently remove a check."""
    policy = {"fairness": {"RAC1P": {"max_tpr_gpa": {"fail": 0.3}}}}
    with pytest.raises(PolicyError, match="max_tpr_gpa"):
        pol.check_fairness(policy, [_summary()])


def test_gating_an_unaudited_attribute_is_fatal():
    policy = {"fairness": {"RELIGION": {"max_tpr_gap": {"fail": 0.3}}}}
    with pytest.raises(PolicyError, match="RELIGION"):
        pol.check_fairness(policy, [_summary()])


def test_every_metric_in_the_real_policy_file_resolves():
    """The shipped policy.yaml must be fully evaluable against a real audit summary.

    This also catches the reverse mistake: declaring a gate for an attribute the audit
    does not produce. It fired when the crossed attribute was added to the policy, which
    is exactly the point of it.
    """
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    summaries = [_summary("RAC1P"), _summary("SEX"), _summary("RACExSEX")]
    checks = pol.check_fairness(policy, summaries)
    declared = sum(len(m) for m in policy["fairness"].values())
    assert len(checks) == declared, "a declared fairness gate did not produce a check"


# --------------------------------------------------------------------------- #
# Direction: ceilings and floors are not the same
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "measured,expected",
    [(0.10, PASS), (0.20, WARN), (0.40, FAIL)],
)
def test_ceiling_metric_direction(measured, expected):
    policy = {"fairness": {"RAC1P": {"max_tpr_gap": {"fail": 0.35, "warn": 0.15}}}}
    checks = pol.check_fairness(policy, [_summary(tpr_gap=measured)])
    assert checks[0].status == expected


@pytest.mark.parametrize(
    "measured,expected",
    [(0.95, PASS), (0.86, WARN), (0.70, FAIL)],
)
def test_floor_metric_direction(measured, expected):
    policy = {"performance": {
        "min_auc": {"fail": 0.85, "warn": 0.88},
        "max_ece": {"fail": 1.0, "warn": 1.0},
        "min_accuracy_over_majority": {"fail": 0.0, "warn": 0.0},
    }}
    scores = {"auc": measured, "ece": 0.01, "accuracy": 0.9}
    checks = pol.check_performance(policy, scores, majority=0.6)
    auc = next(c for c in checks if c.name == "performance.auc")
    assert auc.status == expected


def test_exactly_on_the_threshold_passes():
    """A measured value equal to the limit is not a breach. Stated so it cannot drift."""
    policy = {"fairness": {"RAC1P": {"max_tpr_gap": {"fail": 0.35, "warn": 0.15}}}}
    assert pol.check_fairness(policy, [_summary(tpr_gap=0.15)])[0].status == PASS
    assert pol.check_fairness(policy, [_summary(tpr_gap=0.35)])[0].status == WARN


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def test_one_fail_beats_any_number_of_passes():
    policy = {"fairness": {"RAC1P": {
        "max_tpr_gap": {"fail": 0.05},
        "max_fpr_gap": {"fail": 0.9},
    }}}
    checks = pol.check_fairness(policy, [_summary()])
    assert pol.verdict(checks) == FAIL


def test_warn_does_not_fail_the_build():
    policy = {"fairness": {"RAC1P": {"max_tpr_gap": {"fail": 0.35, "warn": 0.05}}}}
    checks = pol.check_fairness(policy, [_summary(tpr_gap=0.10)])
    assert checks[0].status == WARN
    assert pol.verdict(checks) == WARN


def test_warn_line_quotes_the_target_not_the_tolerance():
    """A WARN that printed the failing limit would read as if it had passed."""
    policy = {"fairness": {"RAC1P": {"max_tpr_gap": {"fail": 0.35, "warn": 0.15}}}}
    line = pol.check_fairness(policy, [_summary(tpr_gap=0.31)])[0].line()
    assert "target <= 0.15" in line
    assert "tolerated to 0.35" in line


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def test_regression_catches_a_drop_that_still_passes_the_floor():
    """The point of a baseline: 0.93 -> 0.89 breaches nothing absolute, but it moved."""
    policy = {"regression": {"max_auc_drop": 0.01, "max_ece_increase": 0.01,
                             "max_gap_increase": 0.02}}
    current = {"auc": 0.89, "ece": 0.02, "gaps": {}}
    baseline = {"auc": 0.93, "ece": 0.02, "gaps": {}}
    checks = pol.check_regression(policy, current, baseline)
    assert next(c for c in checks if c.name == "regression.auc_drop").status == FAIL


def test_an_improvement_is_never_a_regression():
    policy = {"regression": {"max_auc_drop": 0.01, "max_ece_increase": 0.01,
                             "max_gap_increase": 0.02}}
    current = {"auc": 0.95, "ece": 0.01, "gaps": {"RAC1P.tpr_gap": 0.10}}
    baseline = {"auc": 0.90, "ece": 0.03, "gaps": {"RAC1P.tpr_gap": 0.20}}
    assert pol.verdict(pol.check_regression(policy, current, baseline)) == PASS


def test_no_baseline_means_no_regression_checks():
    policy = {"regression": {"max_auc_drop": 0.01, "max_ece_increase": 0.01,
                             "max_gap_increase": 0.02}}
    assert pol.check_regression(policy, {"auc": 0.5, "ece": 0.5}, None) == []


# --------------------------------------------------------------------------- #
# The shipped policy file
# --------------------------------------------------------------------------- #
def test_shipped_policy_has_fail_looser_than_warn_everywhere():
    """warn is the target, fail is the tolerance. Inverting them silently inverts a gate."""
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    for section in ("performance", "fairness"):
        block = policy[section]
        groups = block.values() if section == "fairness" else {"_": block}.values()
        for metrics in groups:
            for metric, bounds in metrics.items():
                if "fail" not in bounds or "warn" not in bounds:
                    continue
                if metric.startswith("max_"):
                    assert bounds["fail"] >= bounds["warn"], metric
                else:
                    assert bounds["fail"] <= bounds["warn"], metric
