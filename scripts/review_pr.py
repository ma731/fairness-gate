"""Compare two audits and write the review comment a pull request should get.

A green build tells you nothing moved past a limit. It does not tell you what moved, in
which direction, or which group absorbed it, and that is the part a reviewer actually
has to think about. A change can keep every check inside its limit and still take four
points of recall off one group, and I did not want to be the person who merged that
because the tick was green.

So this reads the audit on the base branch and the audit on the branch, and writes what
changed in plain language: which checks changed status, which numbers moved more than a
threshold worth mentioning, and which groups are behind the movement.

It makes no judgement the policy does not already make. Passing or failing stays the
gate's job in check_policy.py. This only says what is different, because a reviewer who
can see what moved will ask better questions than one reading a tick.

    python scripts/review_pr.py --base before.json --head after.json
    python scripts/review_pr.py --base before.json --head after.json --out comment.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.glossary import explain

# Below this, a move is rounding and reporting it would train people to skim the
# comment. Rates on 600,000 people are stable to about the third decimal.
NOTABLE = 0.002

STATUS_MARK = {"pass": "pass", "warn": "warn", "fail": "FAIL"}


def _checks(audit: dict) -> dict:
    return {c["name"]: c for c in audit.get("checks", [])}


def _gaps(audit: dict, split: str = "test") -> dict:
    out = {}
    for row in audit.get("fairness", {}).get(split, []):
        for metric in ("tpr_gap", "fpr_gap", "base_rate_gap"):
            if row.get(metric) is not None:
                out[f"{row['attribute']}.{metric}"] = float(row[metric])
    return out


def _arrow(delta: float, higher_is_worse: bool = True) -> str:
    if abs(delta) < NOTABLE:
        return "no change"
    worse = (delta > 0) if higher_is_worse else (delta < 0)
    return f"{delta:+.4f} {'worse' if worse else 'better'}"


def status_changes(base: dict, head: dict) -> list[str]:
    """Checks that crossed a line. The only part of this that is not advisory."""
    b, h = _checks(base), _checks(head)
    lines = []
    for name, check in h.items():
        was = b.get(name, {}).get("status")
        now = check["status"]
        if was is None:
            title, _ = explain(name)
            lines.append(f"- **new check** `{name}` ({title}) is at **{STATUS_MARK[now]}**")
        elif was != now:
            title, _ = explain(name)
            lines.append(
                f"- `{name}` ({title}) went **{STATUS_MARK[was]} to {STATUS_MARK[now]}**"
            )
    for name in b:
        if name not in h:
            lines.append(f"- **check removed**: `{name}` is no longer in the policy")
    return lines


def moved_numbers(base: dict, head: dict) -> list[tuple[str, float, float, float]]:
    """Every measured check that moved by more than the noise floor."""
    b, h = _checks(base), _checks(head)
    moves = []
    for name, check in h.items():
        if name not in b:
            continue
        before, after = b[name].get("measured"), check.get("measured")
        if before is None or after is None:
            continue
        delta = float(after) - float(before)
        if abs(delta) >= NOTABLE:
            moves.append((name, float(before), float(after), delta))
    moves.sort(key=lambda m: -abs(m[3]))
    return moves


def group_movement(base: dict, head: dict, attribute: str = "RAC1P") -> list[str]:
    """Which groups are behind the movement, in people rather than in deltas."""
    lines = []
    for metric, label, worse_up in (
        ("tpr_gap", "recall gap", True),
        ("fpr_gap", "false alarm gap", True),
    ):
        key = f"{attribute}.{metric}"
        before = _gaps(base).get(key)
        after = _gaps(head).get(key)
        if before is None or after is None:
            continue
        delta = after - before
        if abs(delta) < NOTABLE:
            continue
        lines.append(
            f"- the **{label}** across {attribute} moved {before:.4f} to {after:.4f} "
            f"({_arrow(delta, worse_up)})"
        )
    return lines


def render(base: dict, head: dict) -> str:
    """The comment body. Markdown, because that is what a PR renders."""
    status = status_changes(base, head)
    moves = moved_numbers(base, head)
    groups = group_movement(base, head)

    verdict = head.get("policy_verdict", "unknown")
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in head.get("checks", []):
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    out = ["## Fairness review", ""]

    if not status and not moves:
        out += [
            (
                "Nothing measurable changed. Every check holds the same status and no "
                f"number moved by more than {NOTABLE:.3f}, which on this many people "
                "is rounding."
            ),
            "",
        ]
    else:
        out += [
            (
                f"The gate is at **{verdict}** with {counts['pass']} passing, "
                f"{counts['warn']} warning and {counts['fail']} failing. "
                "Here is what actually moved."
            ),
            "",
        ]

    if status:
        out += ["### Checks that crossed a line", "", *status, ""]

    if moves:
        out += [
            "### Numbers that moved",
            "",
            "| check | before | after | change |",
            "|---|---:|---:|---|",
        ]
        for name, before, after, delta in moves[:12]:
            out.append(f"| `{name}` | {before:.4f} | {after:.4f} | {_arrow(delta)} |")
        if len(moves) > 12:
            out.append(f"\n{len(moves) - 12} smaller moves not shown.")
        out.append("")

    if groups:
        out += ["### Who it lands on", "", *groups, ""]

    if moves or status:
        out += [
            "---",
            "",
            (
                "Worth asking before merging: is this change meant to move these "
                "numbers? If it is, the policy limits in `policy.yaml` may need to "
                "move with it, and that belongs in this pull request rather than a "
                "later one. If it is not, something changed the model that was not "
                "supposed to."
            ),
        ]

    out += [
        "",
        (
            "<sub>Posted by `scripts/review_pr.py`. It reports differences and makes "
            "no pass or fail judgement: that stays with the gate.</sub>"
        ),
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="audit.json from the base branch")
    ap.add_argument("--head", required=True, help="audit.json from this branch")
    ap.add_argument("--out", help="write the comment here instead of stdout")
    args = ap.parse_args()

    base_path, head_path = Path(args.base), Path(args.head)
    if not head_path.exists():
        print(f"no audit at {head_path}", file=sys.stderr)
        return 1
    # A base with no audit is normal on the first pull request, and is not an error.
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    head = json.loads(head_path.read_text(encoding="utf-8"))

    body = render(base, head)
    if args.out:
        Path(args.out).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
