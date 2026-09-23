"""Score the narrator's verifier against the drafts in evals/.

Needs no API key and no data. Reports misses and false alarms, since a verifier that
rejects everything would catch every error and still be useless.

    python scripts/run_evals.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from evals.cases import CASES
from src.narrator import facts, verify
from src.report import RESULTS_DIR


def load_facts():
    audit = RESULTS_DIR / "audit.json"
    if not audit.exists():
        print(f"no audit at {audit}. Run scripts/run_audit.py.", file=sys.stderr)
        return None
    result = json.loads(audit.read_text(encoding="utf-8"))
    tables = {
        split: pd.read_csv(RESULTS_DIR / f"groups_{split}.csv")
        for split in ("test", "shift")
    }
    return facts(result, tables)


def run(verbose: bool = False) -> tuple[list[dict], dict]:
    f = load_facts()
    if f is None:
        return [], {}

    rows, missed, false_alarms = [], 0, 0
    for case in CASES:
        raised = {v.code for v in verify(case["text"], f)}
        expected = set(case["expect"])

        # A miss is an expected violation the verifier did not raise. Dangerous.
        not_caught = expected - raised
        # A false alarm is any violation the draft didn't earn, on a bad draft as well as
        # a clean one. Only checking clean drafts missed a real false alarm once.
        spurious = raised - expected

        ok = not not_caught and not spurious
        missed += len(not_caught)
        false_alarms += len(spurious)
        rows.append({
            "name": case["name"],
            "ok": ok,
            "expected": sorted(expected),
            "raised": sorted(raised),
            "missed": sorted(not_caught),
            "spurious": sorted(spurious),
            "why": case["why"],
        })

    summary = {
        "cases": len(rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "missed_violations": missed,
        "false_alarms": false_alarms,
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="show every case")
    args = ap.parse_args()

    rows, summary = run(args.verbose)
    if not rows:
        return 1

    for r in rows:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['name']}")
        if args.verbose or not r["ok"]:
            print(f"         {r['why']}")
            print(f"         expected {r['expected'] or 'nothing'}, "
                  f"raised {r['raised'] or 'nothing'}")
            if r["missed"]:
                print(f"         MISSED: {r['missed']}")
            if r["spurious"]:
                print(f"         FALSE ALARM: {r['spurious']}")

    print(
        f"\n{summary['passed']}/{summary['cases']} cases, "
        f"{summary['missed_violations']} missed, "
        f"{summary['false_alarms']} false alarms"
    )
    passed = summary["passed"] == summary["cases"]
    print("EVALS PASSED" if passed else "EVALS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
