"""The gate. Runs in seconds, needs no data, and is what CI enforces on every commit.

Three things are checked:

1. The committed audit result still satisfies policy.yaml. Loosening a threshold is then
   a visible diff in a file a reviewer has to approve, not a quiet edit.
2. The committed documents are exactly what the committed results generate. This is what
   makes "the model card is generated, not written" a fact rather than a claim: hand-edit
   the model card and this fails.
3. The audit result is not stale relative to the code that produces it.

The full audit lives in scripts/run_audit.py and needs ~3 GB of census data, so it runs
locally and on a schedule, not on every push.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src import policy as pol
from src.dashboard import PAGES as SITE_PAGES
from src.dashboard import render as render_dashboard
from src.fingerprint import pipeline_fingerprint
from src.pages import about as render_about
from src.report import annex_iv, checks_doc, dpia, model_card

AUDIT_PATH = REPO_ROOT / "results" / "audit.json"

# Files that must match what the generator produces, and how to produce them.
GENERATED = {
    REPO_ROOT / "docs" / "model_card.md": lambda r, t: model_card(r, t),
    REPO_ROOT / "docs" / "annex_iv.md": lambda r, t: annex_iv(r, t),
    REPO_ROOT / "docs" / "dpia.md": lambda r, t: dpia(r),
    REPO_ROOT / "results" / "checks.md": lambda r, t: checks_doc(r),
    REPO_ROOT / "docs" / "about.html": lambda r, t: render_about(r),
}

# Every page of the site, added here rather than listed by hand, so a page that gets
# added to the site cannot quietly escape the byte-identical check.
for _name, *_rest in SITE_PAGES:
    GENERATED[REPO_ROOT / "docs" / _name] = (
        lambda r, t, _n=_name: render_dashboard(r, t, _n)
    )

def _fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)


def _load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for split in ("test", "shift"):
        path = REPO_ROOT / "results" / f"groups_{split}.csv"
        if path.exists():
            tables[split] = pd.read_csv(path)
    return tables


def check_thresholds(result: dict) -> bool:
    """Re-evaluate the committed numbers against the current policy file.

    Recomputed rather than trusted: the stored verdict was produced under whatever
    policy.yaml said at the time, and the policy may have changed since.
    """
    policy = pol.load_policy()
    scores = result["scores"]["test"]
    summaries = result["fairness"]["test"]

    race_test = next(s for s in summaries if s["attribute"] == "RAC1P")
    race_shift = next(
        s for s in result["fairness"]["shift"] if s["attribute"] == "RAC1P"
    )

    checks = (
        pol.check_performance(policy, scores, scores["majority_baseline"])
        + pol.check_fairness(policy, summaries, result.get("uncertainty"))
        + pol.check_shift_canary(
            policy, result["scores"]["shift"], race_test["tpr_gap"], race_shift["tpr_gap"]
        )
    )

    for c in checks:
        if c.status != pol.PASS:
            print("  " + c.line())

    print(f"  {pol.summarise(checks)}")

    if pol.verdict(checks) == pol.FAIL:
        _fail("a declared threshold is breached")
        return False
    return True


def check_documents_are_generated(result: dict) -> bool:
    """Regenerate every document and require a byte-identical match."""
    tables = _load_tables()
    if "test" not in tables:
        _fail("results/groups_test.csv missing; cannot verify documents")
        return False

    ok = True
    for path, render in GENERATED.items():
        if not path.exists():
            _fail(f"{path.relative_to(REPO_ROOT)} is missing")
            ok = False
            continue
        expected = render(result, tables)
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            ok = False
            _fail(
                f"{path.relative_to(REPO_ROOT)} does not match what the audit generates. "
                "It was edited by hand, or the results changed without regenerating."
            )
            diff = list(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="generated",
                    tofile="committed",
                    lineterm="",
                    n=1,
                )
            )
            for line in diff[:20]:
                print("    " + line, file=sys.stderr)
            if len(diff) > 20:
                print(f"    ... {len(diff) - 20} more lines", file=sys.stderr)
        else:
            print(f"  ok  {path.relative_to(REPO_ROOT)}")
    return ok


def check_not_stale(result: dict) -> bool:
    """Do these results match the code that is in the tree right now?"""
    recorded = result.get("pipeline_fingerprint")
    if not recorded:
        _fail("audit has no pipeline fingerprint. Re-run scripts/run_audit.py.")
        return False

    current = pipeline_fingerprint()
    if recorded != current:
        _fail(
            f"the pipeline changed since this audit ran (recorded {recorded}, "
            f"current {current}). These results are not evidence about this code. "
            "Re-run scripts/run_audit.py."
        )
        return False
    print(f"  results match the current pipeline ({current})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-staleness", action="store_true",
                    help="skip the pipeline fingerprint check")
    args = ap.parse_args()

    if not AUDIT_PATH.exists():
        _fail(f"{AUDIT_PATH.relative_to(REPO_ROOT)} not found. Run scripts/run_audit.py.")
        return 1

    result = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    print(f"audit generated {result['generated_at']} at {result['git_sha']}\n")

    print("thresholds:")
    ok_thresholds = check_thresholds(result)

    print("\ngenerated documents:")
    ok_docs = check_documents_are_generated(result)

    ok_fresh = True
    if not args.skip_staleness:
        print("\nfreshness:")
        ok_fresh = check_not_stale(result)

    passed = ok_thresholds and ok_docs and ok_fresh
    print("\n" + ("GATE PASSED" if passed else "GATE FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
