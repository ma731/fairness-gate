"""Tests for the generated documents.

These run against the committed results, so they need no census data and work in CI.

The property that matters is determinism: `scripts/check_policy.py` proves the documents
were not hand-edited by regenerating them and requiring a byte-identical match. That
proof is worthless if generation is not deterministic, so it is tested directly.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from src import report
from src.policy import BASELINE_PATH  # noqa: F401  (kept for path discovery in CI)

AUDIT_PATH = report.RESULTS_DIR / "audit.json"

pytestmark = pytest.mark.skipif(
    not AUDIT_PATH.exists(), reason="no committed audit; run scripts/run_audit.py"
)


@pytest.fixture(scope="module")
def result():
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables():
    return {
        split: pd.read_csv(report.RESULTS_DIR / f"groups_{split}.csv")
        for split in ("test", "shift")
    }


@pytest.mark.parametrize("name", ["model_card", "annex_iv"])
def test_generation_is_deterministic(name, result, tables):
    render = getattr(report, name)
    assert render(result, tables) == render(result, tables)


def test_dpia_generation_is_deterministic(result):
    assert report.dpia(result) == report.dpia(result)


@pytest.mark.parametrize("name", ["model_card", "annex_iv"])
def test_documents_say_they_are_generated(name, result, tables):
    """A reader who opens the file must know not to edit it."""
    text = getattr(report, name)(result, tables)
    assert "GENERATED FILE" in text
    assert "Do not edit by hand" in text


def test_model_card_reports_the_real_tpr_gap(result, tables):
    card = report.model_card(result, tables)
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    assert f"{race['tpr_gap']:.3f}" in card


def test_model_card_names_both_ends_of_the_worst_gap(result, tables):
    """The narrative must name groups, not just quote a gap in the abstract."""
    card = report.model_card(result, tables)
    test = tables["test"]
    sub = test[(test["attribute"] == "RAC1P") & test["reportable"]]
    best = sub.loc[sub["tpr"].idxmax(), "group"]
    worst = sub.loc[sub["tpr"].idxmin(), "group"]
    assert best in card and worst in card


def test_model_card_does_not_present_validation_ece_as_a_result(result, tables):
    """The calibrator was fitted on val, so its ECE there measures nothing."""
    card = report.model_card(result, tables)
    assert "not a result" in card


def test_small_groups_are_never_in_a_reported_table(result, tables):
    """Groups under the reporting floor must not appear in any document body."""
    card = report.model_card(result, tables)
    test = tables["test"]
    for _, row in test[~test["reportable"]].iterrows():
        assert f"| {row['group']} |" not in card


def test_annex_iv_records_the_rejected_mitigation(result, tables):
    """A mitigation considered and refused belongs in the documentation, with the why.

    Asserts the substance rather than a turn of phrase: that the document says it was
    rejected, names the reason that actually disqualifies it, and carries the measured
    numbers instead of claiming a cost it never computed.
    """
    text = report.annex_iv(result, tables).lower()
    assert "not adopted" in text or "not applied" in text
    assert "moment of prediction" in text or "inference time" in text

    mit = (result.get("mitigation") or {}).get("RAC1P") or {}
    if mit:
        # the numbers, not an adjective
        assert f"{mit['per_group']['tpr_gap']:.4f}" in text
        assert f"{mit['accuracy_cost']:.4f}" in text


def test_dpia_states_the_residual_risk_is_unresolved(result):
    text = report.dpia(result)
    assert "not resolved" in text.lower()


def test_every_number_in_the_model_card_comes_from_the_audit(result, tables):
    """No figure may be invented by the template.

    Every decimal in the document must be traceable to the audit JSON, to a percentage
    derived from it, or to the small set of structural constants the template states.
    """
    card = report.model_card(result, tables)
    blob = json.dumps(result) + tables["test"].to_json()

    allowed = {"50,000", "500", "0.5", "1.0"}  # thresholds the text names in prose
    suspicious = []
    for token in re.findall(r"\d+\.\d{3,4}", card):
        if token in allowed:
            continue
        value = float(token)
        # Present in the audit, or a rounding of something in it.
        if token in blob:
            continue
        if any(
            abs(value - float(m)) < 5e-4
            for m in re.findall(r"-?\d+\.\d+", blob)
        ):
            continue
        suspicious.append(token)

    assert not suspicious, f"numbers in the model card not traceable to the audit: {suspicious}"
