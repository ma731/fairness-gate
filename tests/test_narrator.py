"""Tests for the narrator and its verifier.

The eval suite in `evals/` scores the verifier against drafts with planted errors. This
file covers the machinery around it: that the retry loop actually retries, that it tells
the model what was wrong, and above all that it publishes nothing when nothing verifies.

That last one is the whole design. A guardrail that falls back to "the best of four bad
drafts" is not a guardrail, it is a delay.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from evals.cases import CASES
from src import narrator, report

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


@pytest.fixture(scope="module")
def f(result, tables):
    return narrator.facts(result, tables)


def case(name: str) -> dict:
    return next(c for c in CASES if c["name"] == name)


# --- the verifier, through the eval fixtures --------------------------------------- #

@pytest.mark.parametrize("spec", CASES, ids=lambda c: c["name"])
def test_every_eval_case_behaves(spec, f):
    """The suite runs as tests too, so a broken guardrail fails the ordinary test run."""
    raised = {v.code for v in narrator.verify(spec["text"], f)}
    expected = set(spec["expect"])
    assert not (expected - raised), f"missed {expected - raised}: {spec['why']}"
    assert not (raised - expected), f"false alarm {raised - expected}: {spec['why']}"


def test_a_clean_draft_publishes(f):
    assert narrator.verify(case("clean")["text"], f) == []


# --- the number check -------------------------------------------------------------- #

def test_precision_travels_with_the_number(f):
    """0.312 is a rounding of 0.311919. 0.3125 is not, and claims more than it has."""
    assert narrator._supported(0.312, 3, False, {0.311919})
    assert not narrator._supported(0.3125, 4, False, {0.311919})


def test_a_bare_small_decimal_is_not_read_as_a_percentage(f):
    """Reading every number as a percentage too is how the first version leaked.

    It let 0.447 match an unrelated value four decimal places away, because dividing
    by a hundred makes everything small enough to hit the absolute tolerance.
    """
    assert not narrator._supported(0.447, 3, False, {0.00447})


def test_a_percentage_sign_is_honoured(f):
    assert narrator._supported(54.0, 0, True, {0.5420})


def test_the_fact_set_excludes_the_sweep(result, tables):
    """A summary quoting one point off an internal sweep is not summarising.

    Including the sweep made the allowed set so large that three planted errors passed.
    """
    f = narrator.facts(result, tables)
    sweep = (result.get("mitigation") or {}).get("RAC1P", {}).get("sweep") or []
    assert sweep, "no sweep in the audit; this test is no longer checking anything"
    interior = [round(float(p["accuracy"]), 6) for p in sweep[5:-5]]
    assert any(v not in f.numbers for v in interior)


# --- the retry loop ---------------------------------------------------------------- #

def test_it_stops_as_soon_as_a_draft_verifies(f, result, tables):
    calls = []

    def call(system, user):
        calls.append(user)
        return case("clean")["text"]

    out = narrator.generate(result, tables, call)
    assert out["verified"] is True
    assert out["attempts"] == 1
    assert len(calls) == 1


def test_it_tells_the_model_what_was_wrong(f, result, tables):
    """Retrying with the same prompt is just rolling the dice again."""
    seen = []

    def call(system, user):
        seen.append(user)
        return (case("clean")["text"] if len(seen) > 1
                else case("invented_number")["text"])

    out = narrator.generate(result, tables, call)
    assert out["verified"] is True
    assert out["attempts"] == 2
    assert "rejected by the verifier" in seen[1]
    assert "unsupported_number" in seen[1]


def test_nothing_publishes_when_nothing_verifies(f, result, tables):
    """The point of the whole module. No best-effort fallback, no least-bad draft."""
    out = narrator.generate(
        result, tables, lambda s, u: case("claims_the_model_is_fair")["text"]
    )
    assert out["verified"] is False
    assert out["text"] == ""
    assert out["attempts"] == narrator.MAX_ATTEMPTS
    assert len(out["history"]) == narrator.MAX_ATTEMPTS


def test_the_history_records_every_rejected_draft(f, result, tables):
    """A refusal that does not say what it saw is not auditable."""
    out = narrator.generate(
        result, tables, lambda s, u: case("invents_an_attribute")["text"], attempts=2
    )
    assert all(h["violations"] for h in out["history"])
    assert "absent_attribute" in out["history"][0]["violations"][0]


# --- the prompt -------------------------------------------------------------------- #

def test_the_prompt_carries_the_refusal(result, tables):
    """The model is told the mitigation was refused, so it has no excuse to say it shipped."""
    text = narrator.prompt(result, tables)
    assert "NOT adopted" in text
    assert "requires the person's race" in text


def test_the_prompt_states_the_real_numbers(result, tables):
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    assert f"{race['tpr_gap']:.4f}" in narrator.prompt(result, tables)


# --- loading ----------------------------------------------------------------------- #

def test_an_unverified_record_never_loads(tmp_path, monkeypatch):
    path = tmp_path / "narration.json"
    path.write_text(json.dumps({"text": "anything", "verified": False}), encoding="utf-8")
    monkeypatch.setattr(narrator, "NARRATION_PATH", path)
    assert narrator.load() == {}


def test_a_missing_record_is_a_normal_state(tmp_path, monkeypatch):
    monkeypatch.setattr(narrator, "NARRATION_PATH", tmp_path / "nope.json")
    assert narrator.load() == {}
