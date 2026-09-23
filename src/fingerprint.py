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

# Changing any of these changes the numbers. policy.yaml is deliberately NOT here: the
# policy is what the results are judged against, not what produces them.
PIPELINE_SOURCES = [
    "src/config.py",
    "src/data.py",
    "src/model.py",
    "src/fairness.py",
    "src/distributions.py",
    "src/uncertainty.py",
    "src/dashboard.py",
    "src/pages.py",
    "src/theme.py",
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
