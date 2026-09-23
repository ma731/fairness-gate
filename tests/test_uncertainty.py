"""Tests for the intervals, and for the gate refusing to fire on noise.

The behaviour that matters: a build that goes red on sampling noise is a build people
learn to re-run until it passes, which is a gate that has already stopped working. So a
FAIL now requires the whole interval to clear the limit, and a point estimate over the
limit with an interval straddling it is a WARN.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import policy as pol
from src.policy import FAIL, PASS, WARN
from src.uncertainty import gap_interval, wilson


# --------------------------------------------------------------------------- #
# Wilson
# --------------------------------------------------------------------------- #
def test_wilson_narrows_as_the_group_grows():
    """The whole point: the same rate on more people is a stronger claim."""
    small = wilson(69, 100)
    large = wilson(6900, 10000)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 5


def test_wilson_stays_inside_zero_and_one():
    """Where the normal interval escapes the unit interval, Wilson does not."""
    lo, hi = wilson(0, 30)
    assert lo >= 0.0 and hi <= 1.0
    lo, hi = wilson(30, 30)
    assert lo >= 0.0 and hi <= 1.0


def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(180, 400)
    assert lo < 0.45 < hi


def test_wilson_on_an_empty_group_is_not_a_number():
    lo, hi = wilson(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# Gap interval
# --------------------------------------------------------------------------- #
def test_gap_interval_brackets_its_point_estimate():
    trials = [4000, 4000, 4000]
    successes = [3400, 2800, 2000]        # rates .85 .70 .50, gap .35
    ci = gap_interval(successes, trials, draws=2000)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert abs(ci["point"] - 0.35) < 1e-9


def test_a_gap_on_tiny_groups_is_wider_than_the_same_gap_on_large_ones():
    tiny = gap_interval([85, 50], [100, 100], draws=2000)
    huge = gap_interval([85000, 50000], [100000, 100000], draws=2000)
    assert (tiny["hi"] - tiny["lo"]) > (huge["hi"] - huge["lo"]) * 10


def test_gap_interval_is_deterministic():
    a = gap_interval([340, 280, 200], [400, 400, 400], draws=1500)
    b = gap_interval([340, 280, 200], [400, 400, 400], draws=1500)
    assert a == b


def test_gap_interval_needs_two_groups():
    assert gap_interval([50], [100]) == {}


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def _summary(tpr_gap: float):
    return [{
        "attribute": "RAC1P", "tpr_gap": tpr_gap, "fpr_gap": 0.0,
        "calibration_gap": 0.0, "demographic_parity_difference": 0.0,
        "equalized_odds_difference": 0.0, "base_rate_gap": 0.0,
        "accuracy_gap": 0.0, "demographic_parity_ratio": 1.0,
        "groups_reported": 3, "groups_suppressed": 0,
    }]


POLICY = {"fairness": {"RAC1P": {"max_tpr_gap": {"fail": 0.35, "warn": 0.15}}}}


def test_a_breach_the_interval_does_not_confirm_is_only_a_warning():
    """0.36 measured, but the interval straddles the 0.35 limit. Not yet certain."""
    ci = {"RAC1P": {"tpr_gap": {"lo": 0.31, "hi": 0.41, "level": 0.95}}}
    c = pol.check_fairness(POLICY, _summary(0.36), ci)[0]
    assert c.status == WARN
    assert "not yet distinguishable from noise" in c.note


def test_a_breach_the_whole_interval_clears_still_fails():
    """0.42 measured and the interval sits entirely above the limit. Real."""
    ci = {"RAC1P": {"tpr_gap": {"lo": 0.38, "hi": 0.46, "level": 0.95}}}
    c = pol.check_fairness(POLICY, _summary(0.42), ci)[0]
    assert c.status == FAIL


def test_the_interval_never_rescues_a_pass_into_a_failure():
    """Uncertainty may soften a FAIL, never manufacture one."""
    ci = {"RAC1P": {"tpr_gap": {"lo": 0.05, "hi": 0.40, "level": 0.95}}}
    c = pol.check_fairness(POLICY, _summary(0.10), ci)[0]
    assert c.status == PASS


def test_without_an_interval_the_gate_behaves_as_before():
    """No interval exported means the old point-estimate behaviour, not a crash."""
    c = pol.check_fairness(POLICY, _summary(0.42), None)[0]
    assert c.status == FAIL


def test_a_passing_check_still_reports_its_interval():
    ci = {"RAC1P": {"tpr_gap": {"lo": 0.08, "hi": 0.12, "level": 0.95}}}
    c = pol.check_fairness(POLICY, _summary(0.10), ci)[0]
    assert "95% CI" in c.note


@pytest.mark.parametrize("gap,expected", [(0.10, PASS), (0.20, WARN), (0.50, FAIL)])
def test_thresholds_still_apply_when_the_interval_is_tight(gap, expected):
    ci = {"RAC1P": {"tpr_gap": {"lo": gap - 0.005, "hi": gap + 0.005, "level": 0.95}}}
    assert pol.check_fairness(POLICY, _summary(gap), ci)[0].status == expected
