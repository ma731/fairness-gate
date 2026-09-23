"""Tests for the pull request reviewer.

The reviewer runs on other people's branches and writes into a conversation, so the
failure modes that matter are not crashes. They are saying nothing when something moved,
and saying something when nothing did. Both are tested here.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "review_pr", REPO_ROOT / "scripts" / "review_pr.py"
)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def audit(tpr_gap: float = 0.31, status: str = "warn", verdict: str = "warn") -> dict:
    return {
        "policy_verdict": verdict,
        "checks": [
            {"name": "fairness.RAC1P.max_tpr_gap", "status": status,
             "measured": tpr_gap, "warn_at": 0.15, "fail_at": 0.35, "direction": "max"},
            {"name": "performance.auc", "status": "pass",
             "measured": 0.888, "warn_at": 0.85, "fail_at": 0.8, "direction": "min"},
        ],
        "fairness": {"test": [{"attribute": "RAC1P", "tpr_gap": tpr_gap,
                               "fpr_gap": 0.14, "base_rate_gap": 0.27}]},
    }


def test_silence_when_nothing_moved():
    body = review.render(audit(), audit())
    assert "Nothing measurable changed" in body
    assert "Numbers that moved" not in body


def test_rounding_is_not_reported():
    """Below the noise floor, reporting a move teaches people to skim the comment."""
    body = review.render(audit(0.3100), audit(0.3108))
    assert "Nothing measurable changed" in body


def test_a_real_move_is_reported_with_direction():
    body = review.render(audit(0.3100), audit(0.3600))
    assert "Numbers that moved" in body
    assert "+0.0500 worse" in body


def test_an_improvement_is_not_called_worse():
    body = review.render(audit(0.3600), audit(0.3100))
    assert "-0.0500 better" in body
    table = body.split("### Numbers that moved")[1].split("###")[0]
    assert "worse" not in table


def test_a_status_change_is_called_out_by_name():
    body = review.render(audit(0.31, "warn"), audit(0.36, "fail", "fail"))
    assert "warn to FAIL" in body
    assert "Recall gap, racial groups" in body, "the plain title has to travel with it"


def test_a_removed_check_is_noticed():
    """Deleting a check is the quietest way to make a gate pass. It gets said out loud."""
    before = audit()
    after = copy.deepcopy(before)
    after["checks"] = [c for c in after["checks"]
                       if c["name"] != "fairness.RAC1P.max_tpr_gap"]
    body = review.render(before, after)
    assert "check removed" in body
    assert "fairness.RAC1P.max_tpr_gap" in body


def test_the_first_pull_request_does_not_crash():
    """No audit on the base branch is normal the first time, not an error."""
    body = review.render({}, audit())
    assert "new check" in body


def test_it_never_claims_to_pass_or_fail_anything():
    """Judgement belongs to the gate. A bot that both nags and blocks gets muted."""
    body = review.render(audit(0.31), audit(0.36, "fail", "fail"))
    assert "makes no pass or fail judgement" in body


@pytest.mark.parametrize("delta", [0.0, 0.0005, -0.0019])
def test_noise_floor_boundary(delta):
    assert review._arrow(delta) == "no change"
