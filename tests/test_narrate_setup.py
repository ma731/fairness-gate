"""Tests for the narrator's .env setup. No network: only parsing and error messages."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("narrate", REPO_ROOT / "scripts" / "narrate.py")
narrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(narrate)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """A real exported key on the test machine must not leak into these tests."""
    for key in ("NARRATOR_PROVIDER", "NARRATOR_API_KEY", "NARRATOR_MODEL",
                "AZURE_ENDPOINT", "AZURE_RESOURCE", "AZURE_API_VERSION"):
        monkeypatch.delenv(key, raising=False)


def test_env_file_is_parsed_with_comments_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\nNARRATOR_PROVIDER=azure-openai  # inline\n"
        'NARRATOR_MODEL="my-deployment"\n\nAZURE_API_VERSION=\n',
        encoding="utf-8",
    )
    values = narrate.load_env(env)
    assert values["NARRATOR_PROVIDER"] == "azure-openai"
    assert values["NARRATOR_MODEL"] == "my-deployment"
    assert values["AZURE_API_VERSION"] == ""


def test_a_real_environment_variable_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("NARRATOR_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("NARRATOR_MODEL", "from-shell")
    assert narrate.load_env(env)["NARRATOR_MODEL"] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert narrate.load_env(tmp_path / "nope.env") == {}


def test_an_empty_key_says_where_to_find_it():
    with pytest.raises(narrate.SetupError, match="Keys and Endpoint"):
        narrate.make_caller({"NARRATOR_PROVIDER": "azure-openai"})


def test_an_unknown_provider_lists_the_options():
    with pytest.raises(narrate.SetupError, match="azure-openai"):
        narrate.make_caller({"NARRATOR_PROVIDER": "gemini", "NARRATOR_API_KEY": "x"})


def test_azure_needs_a_deployment_name():
    pytest.importorskip("openai")
    with pytest.raises(narrate.SetupError, match="deployment name"):
        narrate.make_caller({
            "NARRATOR_PROVIDER": "azure-openai",
            "NARRATOR_API_KEY": "x",
            "AZURE_ENDPOINT": "https://example.openai.azure.com/",
        })


@pytest.mark.parametrize(("error_name", "hint"), [
    ("AuthenticationError", "key was rejected"),
    ("NotFoundError", "deployment name"),
    ("APIConnectionError", "Could not reach"),
    ("RateLimitError", "Rate limited"),
])
def test_sdk_errors_become_plain_advice(error_name, hint):
    error = type(error_name, (Exception,), {})("boom")
    assert hint in narrate.explain_failure(error)


def test_the_example_file_holds_no_secret():
    """The template is committed. It must never carry a real key or a real resource."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("NARRATOR_API_KEY="))
    assert line == "NARRATOR_API_KEY="
    assert "your-resource-name" in text


def test_a_failed_run_never_replaces_the_published_summary(tmp_path, monkeypatch):
    published = tmp_path / "narration.json"
    published.write_text('{"text": "good", "verified": true}', encoding="utf-8")
    monkeypatch.setattr(narrate, "NARRATION_PATH", published)
    monkeypatch.setattr(narrate, "REJECTED_PATH", tmp_path / "narration_rejected.json")
    monkeypatch.setattr(narrate, "REPO_ROOT", tmp_path)

    narrate.write_record({"text": "", "verified": False})
    assert '"good"' in published.read_text(encoding="utf-8")
    assert (tmp_path / "narration_rejected.json").exists()

    narrate.write_record({"text": "better", "verified": True})
    assert '"better"' in published.read_text(encoding="utf-8")
