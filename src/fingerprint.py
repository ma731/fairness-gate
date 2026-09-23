"""A content hash of the code that determines what an audit produces.

Git history was the first attempt and it was wrong: it only sees committed changes, so
an audit committed alongside its own code always looked stale, and an uncommitted edit
looked clean. Hashing the files themselves answers the question actually being asked,
which is whether these results could have come from this code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only the code that produces a number belongs here.
#
# Two things are deliberately left out. policy.yaml, because the policy is what the
# results get judged against, not what makes them. And everything that draws the site
# (dashboard, pages, theme), because a colour or a sentence cannot move a figure, and
# the documents are already guarded by a stronger check: check_policy.py regenerates
# every one of them and requires a byte-identical match, so changing how something is
# rendered without re-rendering it fails immediately and for the right reason.
#
# Having presentation in here made every cosmetic edit mark the results stale and
# demand a three gigabyte re-run to fix, which taught me to reach for --skip-staleness.
# A check people routinely switch off has stopped being a check.
PIPELINE_SOURCES = [
    "src/config.py",
    "src/data.py",
    "src/model.py",
    "src/fairness.py",
    "src/distributions.py",
    "src/uncertainty.py",
    "src/mitigation.py",
    "scripts/run_audit.py",
]

CRLF = b"\r\n"
LF = b"\n"


def pipeline_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in sorted(PIPELINE_SOURCES):
        path = REPO_ROOT / rel
        h.update(rel.encode())
        # Normalise line endings so a Windows checkout and a Linux CI runner agree.
        h.update(path.read_bytes().replace(CRLF, LF) if path.exists() else b"")
    return h.hexdigest()[:16]
