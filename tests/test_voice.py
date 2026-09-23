"""Tests for the voice agent.

Checks that every number it can say comes from the audit, and that common questions
route to the right answer. The grammar runs as JavaScript in the browser, but the
patterns are simple enough that Python's re agrees with it.
"""

from __future__ import annotations

import hashlib
import json
import re

import pandas as pd
import pytest

from src import report, voice

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
def answers(result, tables):
    return voice._answers(result, tables)


def route(said: str) -> str:
    """The same first-match-wins walk the browser does."""
    for pattern, key in voice.GRAMMAR:
        if re.search(pattern, said, re.IGNORECASE):
            return key
    return "unknown"


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("what is a CI gate", "gate"),
        ("what did you find", "finding"),
        ("how many people are overlooked", "people"),
        ("what does recall mean", "recall"),
        ("can you fix it", "fix"),
        ("why not", "why"),
        ("why did you reject the mitigation", "why"),
        ("what happens at the intersections", "intersection"),
        ("how sure are you", "certainty"),
        ("where does the data come from", "data"),
        ("does the model see race", "protected"),
        ("is the build passing", "status"),
        ("what about the small groups", "small"),
        ("help", "help"),
        ("what is your favourite colour", "unknown"),
    ],
)
def test_questions_reach_the_right_answer(said, expected):
    assert route(said) == expected


def test_refusal_beats_the_fix(answers):
    """Order matters more than any single pattern.

    "why did you not fix it" contains the word fix. If the broad rule ran first the
    agent would cheerfully describe the mitigation it refused to ship, which is the one
    answer that would misrepresent the project.
    """
    assert route("why did you not fix it") == "why"
    assert "not ship" in answers["fix"]


def test_every_grammar_key_has_an_answer(answers):
    for _, key in voice.GRAMMAR:
        assert key in answers, f"grammar routes to {key!r} with nothing to say"
    assert "unknown" in answers


def test_the_agent_admits_when_it_does_not_know(answers):
    assert "do not know" in answers["unknown"]


def test_no_answer_is_empty(answers):
    for key, said in answers.items():
        assert len(said.split()) > 12, f"{key} is too thin to be worth speaking"


def test_every_number_the_agent_speaks_comes_from_the_audit(answers, result, tables):
    """The whole justification for building it this way.

    If the agent can say a figure that is not in the audit, it is just a chatbot with a
    nice story attached.
    """
    blob = json.dumps(result) + tables["test"].to_json()
    spoken = " ".join(answers.values())

    # One figure is a sum rather than a stored value: the total overlooked across every
    # reported group. Derived is not invented, but "derived" is exactly the excuse a
    # wrong number would hide behind, so it gets checked against its source instead of
    # waved through.
    overlooked = round(float(voice.human_cost(tables["test"])["overlooked"].sum()))
    assert f"{overlooked:,}" in answers["people"]
    blob += f" {overlooked} "

    # Counts and populations are rendered with separators; strip them before matching.
    suspicious = []
    for token in re.findall(r"\d[\d,]*\.?\d*", spoken):
        plain = token.replace(",", "")
        if plain in blob:
            continue
        value = float(plain)
        if value == int(value) and str(int(value)) in blob:
            continue
        # a rounding of something measured, or a small integer the prose spells out
        if any(
            abs(value - float(m)) < 5e-4 for m in re.findall(r"-?\d+\.\d+", blob)
        ):
            continue
        if value < 1000 and value == int(value):
            continue  # counts of checks, groups, years, percentages in plain words
        suspicious.append(token)

    assert not suspicious, f"the agent can say numbers the audit does not contain: {suspicious}"


def test_the_panel_discloses_where_the_audio_goes(result, tables):
    """Chrome ships the audio off to a speech service. Saying so is not optional here."""
    markup = voice.build(result, tables)
    assert "sends the audio" in markup


def test_it_still_works_without_a_microphone(result, tables):
    """Firefox and Safari have no recognition, so the text box has to carry it."""
    markup = voice.build(result, tables)
    assert "cannot do speech recognition" in markup
    assert 'id="vx-form"' in markup


def test_build_is_deterministic(result, tables):
    """It lands in index.html, which is compared byte for byte."""
    assert voice.build(result, tables) == voice.build(result, tables)


def test_the_agent_does_not_claim_the_model_is_blind_to_race(answers):
    """I shipped that claim once. It was false, and it is the kind of false that a
    reviewer checks first, so it gets a test rather than a promise."""
    said = answers["protected"].lower()
    assert "never" not in said
    assert "it does see them" in said


def test_the_unawareness_answer_quotes_the_experiment(answers, result):
    c = (result.get("unaware") or {}).get("comparison") or {}
    assert c, "no unaware arm in the audit"
    assert f"{c['tpr_gap_unaware']:.4f}" in answers["protected"]


def test_the_agent_picks_an_english_voice(result, tables):
    """The synthesis default is the machine's locale. On a China-locale Windows that
    meant a Chinese voice reading American census figures, which sounds broken."""
    markup = voice.build(result, tables)
    assert "VOICE_RANK" in markup
    assert "u.voice = picked" in markup
    assert "onvoiceschanged" in markup


def test_clips_are_played_from_files_not_synthesised(result, tables):
    """A hosted speech API would need a key, and a key in a public page is published."""
    markup = voice.build(result, tables)
    assert "new Audio(file)" in markup
    assert "clipFor" in markup


def test_a_stale_clip_is_refused(result, tables):
    """Audio confidently reading a number the page has moved past is the worst case.

    The clip records the hash of the text it was rendered from. If the answer changed
    and nobody re-rendered, the hashes differ and the browser voice takes over.
    """
    markup = voice.build(result, tables)
    assert "clip.sha === DATA.hashes[key]" in markup


def test_every_committed_clip_matches_its_answer(result, tables):
    """If this fails, run scripts/voice_audio.py. It means the audio is out of date."""
    clips = voice._audio()
    if not clips:
        pytest.skip("no clips rendered yet")
    answers = voice._answers(result, tables)
    stale = [
        key for key, clip in clips.items()
        if key in answers
        and clip["sha"] != hashlib.sha256(answers[key].encode("utf-8")).hexdigest()[:16]
    ]
    assert not stale, f"stale audio for {stale}; re-run scripts/voice_audio.py"


def test_synthesis_is_only_the_fallback(result, tables):
    markup = voice.build(result, tables)
    assert "function synthesise(" in markup
    assert "player = null; synthesise(what)" in markup
