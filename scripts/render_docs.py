"""Rebuild every document from the committed audit, without the census data.

For wording and design changes. It can't change a number; for that, run the audit.

    python scripts/render_docs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.report import RESULTS_DIR, write_all


def main() -> int:
    audit = RESULTS_DIR / "audit.json"
    if not audit.exists():
        print(f"{audit} not found. Run scripts/run_audit.py first.", file=sys.stderr)
        return 1

    result = json.loads(audit.read_text(encoding="utf-8"))
    tables = {
        split: pd.read_csv(RESULTS_DIR / f"groups_{split}.csv")
        for split in ("test", "shift")
    }

    write_all(result, tables)
    print(f"documents rebuilt from the audit of {result['generated_at']}")
    print("run scripts/check_policy.py to confirm they still match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
