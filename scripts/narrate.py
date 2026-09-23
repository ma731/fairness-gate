"""Write the summary, verify it, and publish it only if it passes.

    python scripts/narrate.py --check                # one tiny call to test the setup
    python scripts/narrate.py                        # generate, verify, retry
    python scripts/narrate.py --from-file draft.md   # verify a draft written elsewhere

Keys live in a local .env (copy .env.example), which git ignores. Any provider works:
the verifier rejects a bad draft whatever wrote it, so a cheaper model only costs retries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.narrator import NARRATION_PATH, facts, generate, prompt, verify
from src.report import RESULTS_DIR

ENV_PATH = REPO_ROOT / ".env"
REJECTED_PATH = NARRATION_PATH.with_name("narration_rejected.json")
PROVIDERS = ("azure-openai", "azure-claude", "anthropic")


class SetupError(Exception):
    """Something in .env is missing or wrong. The message says what to fix."""


def load_env(path: Path = ENV_PATH) -> dict:
    """KEY=value lines from .env, with comments and blanks skipped.

    Real environment variables win, so CI or a shell export can override the file.
    """
    values: dict = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(values) + ["NARRATOR_PROVIDER", "NARRATOR_API_KEY", "NARRATOR_MODEL",
                               "AZURE_ENDPOINT", "AZURE_RESOURCE", "AZURE_API_VERSION"]:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def _need(env: dict, key: str, hint: str) -> str:
    value = env.get(key, "")
    if not value:
        raise SetupError(f"{key} is empty in .env. {hint}")
    return value


def make_caller(env: dict):
    """Build `call(system, user) -> str` for whichever provider .env names."""
    provider = env.get("NARRATOR_PROVIDER", "")
    if provider not in PROVIDERS:
        raise SetupError(f"NARRATOR_PROVIDER must be one of {', '.join(PROVIDERS)}.")
    key = _need(env, "NARRATOR_API_KEY", "Paste KEY 1 from Azure > Keys and Endpoint.")

    if provider == "azure-openai":
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise SetupError("Run: pip install openai") from e
        client = AzureOpenAI(
            azure_endpoint=_need(env, "AZURE_ENDPOINT", "Copy the Endpoint from Azure."),
            api_key=key,
            api_version=env.get("AZURE_API_VERSION") or "2024-10-21",
        )
        # On Azure the model is addressed by the deployment name you chose, not its id.
        deployment = _need(env, "NARRATOR_MODEL", "Put your deployment name here.")

        def call(system: str, user: str) -> str:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return response.choices[0].message.content or ""

        return call

    try:
        import anthropic
    except ImportError as e:
        raise SetupError("Run: pip install anthropic") from e

    model = env.get("NARRATOR_MODEL") or "claude-opus-5"
    if provider == "azure-claude":
        client = anthropic.AnthropicFoundry(
            api_key=key,
            resource=_need(env, "AZURE_RESOURCE", "Your Azure resource name."),
        )
        extra: dict = {}
    else:
        client = anthropic.Anthropic(api_key=key)
        # Server-side fallback reruns a declined request on another model. It exists on
        # the Claude API only, not on Azure.
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    def call(system: str, user: str) -> str:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            **extra,
        )
        if response.stop_reason == "refusal":
            # An empty draft fails the verifier, so the loop simply tries again.
            print("  the model declined this attempt", file=sys.stderr)
            return ""
        return "".join(b.text for b in response.content if b.type == "text")

    return call


def explain_failure(error: Exception) -> str:
    """Turn an SDK error into what to change in .env."""
    name = type(error).__name__
    if "Authentication" in name or "PermissionDenied" in name:
        return "The key was rejected. Re-copy KEY 1 from Azure > Keys and Endpoint."
    if "NotFound" in name:
        return ("Endpoint reached, but no model by that name. NARRATOR_MODEL must be the "
                "deployment name shown in Azure AI Foundry > Deployments.")
    if "Connection" in name or "Timeout" in name:
        return "Could not reach the endpoint. Check AZURE_ENDPOINT (or AZURE_RESOURCE)."
    if "RateLimit" in name:
        return "Rate limited. Wait a minute and try again."
    return f"{name}: {error}"


def load_audit() -> tuple[dict, dict]:
    audit = RESULTS_DIR / "audit.json"
    if not audit.exists():
        raise SystemExit(f"no audit at {audit}. Run scripts/run_audit.py first.")
    result = json.loads(audit.read_text(encoding="utf-8"))
    tables = {
        split: pd.read_csv(RESULTS_DIR / f"groups_{split}.csv")
        for split in ("test", "shift")
    }
    return result, tables


def write_record(record: dict) -> None:
    """A passing draft replaces the published one. A failed run is kept aside for
    inspection, so a bad retry can't wipe a summary that already passed."""
    path = NARRATION_PATH if record.get("verified") else REJECTED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="test the .env setup and stop")
    ap.add_argument("--from-file", help="verify a draft from this file instead")
    ap.add_argument("--author", default="", help="who or what wrote a --from-file draft")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    result, tables = load_audit()
    if args.show_prompt:
        print(prompt(result, tables))
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit_at = result.get("generated_at", "")

    if args.from_file:
        draft = Path(args.from_file).read_text(encoding="utf-8").strip()
        violations = verify(draft, facts(result, tables))
        for v in violations:
            print(f"  rejected: {v}")
        write_record({
            "text": draft if not violations else "",
            "verified": not violations,
            "attempts": 1,
            "source": args.author or f"drafted by hand, from {args.from_file}",
            "model": None,
            "generated_at": now,
            "audit_generated_at": audit_at,
            "history": [{"attempt": 1, "draft": draft,
                         "violations": [str(v) for v in violations]}],
        })
        print("VERIFIED" if not violations
              else "REJECTED; the published summary is unchanged")
        return 0 if not violations else 1

    env = load_env()
    try:
        call = make_caller(env)
        if args.check:
            reply = call("Reply with the single word: ok", "ping")
            print(f"setup works ({env['NARRATOR_PROVIDER']}), the model replied: "
                  f"{reply.strip()[:40]!r}")
            return 0
        out = generate(result, tables, call)
    except SetupError as e:
        print(f"setup: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001  any SDK error gets a plain-language hint
        print(f"call failed: {explain_failure(e)}", file=sys.stderr)
        return 1

    for entry in out["history"]:
        if entry["violations"]:
            print(f"  attempt {entry['attempt']} rejected:")
            for v in entry["violations"]:
                print(f"    {v}")

    model = env.get("NARRATOR_MODEL") or "claude-opus-5"
    write_record({
        **out,
        "source": f"generated by {model} via {env['NARRATOR_PROVIDER']}, "
                  "verified by src/narrator.py",
        "model": model,
        "generated_at": now,
        "audit_generated_at": audit_at,
    })
    print(f"VERIFIED after {out['attempts']} attempt(s)" if out["verified"]
          else f"REJECTED after {out['attempts']} attempts; the published summary "
               "is unchanged")
    return 0 if out["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
