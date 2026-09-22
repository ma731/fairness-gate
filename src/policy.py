"""Turn policy.yaml into pass/fail checks against a completed audit.

Nothing here computes a metric. It only compares measured numbers to declared ones, so
the thresholds stay in a file a non-engineer can read and argue with, and the code stays
a dumb comparator. That separation is the whole point: if a gate loosens, it loosens in a
diff someone has to approve, not in a function nobody reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policy.yaml"
BASELINE_PATH = REPO_ROOT / "results" / "baseline.json"

class PolicyError(ValueError):
    """A declared gate that cannot be evaluated. Always fatal: a gate that silently
    does not run is worse than no gate, because it looks like coverage."""


PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    """One declared expectation, and what actually happened."""

    name: str
    metric: str
    measured: float
    fail_at: float | None
    warn_at: float | None
    direction: str  # "max" = lower is better, "min" = higher is better
    status: str
    note: str = ""

    def line(self) -> str:
        """Report the limit that this status is about, not whichever one exists.

        A WARN that quotes the failing threshold reads as if it passed.
        """
        mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}[self.status]
        rel = "<=" if self.direction == "max" else ">="
        if self.status == FAIL:
            return f"[{mark}] {self.name}: {self.measured:.4f} (limit {rel} {self.fail_at})"
        if self.status == WARN:
            return (f"[{mark}] {self.name}: {self.measured:.4f} "
                    f"(target {rel} {self.warn_at}, tolerated to {self.fail_at})")
        return f"[{mark}] {self.name}: {self.measured:.4f}"


def load_policy(path: Path | None = None) -> dict:
    with open(path or POLICY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _evaluate(name, metric, measured, bounds, direction) -> Check:
    fail_at = bounds.get("fail")
    warn_at = bounds.get("warn")

    def breaches(limit):
        if limit is None:
            return False
        return measured > limit if direction == "max" else measured < limit

    status = FAIL if breaches(fail_at) else (WARN if breaches(warn_at) else PASS)
    return Check(
        name=name,
        metric=metric,
        measured=float(measured),
        fail_at=fail_at,
        warn_at=warn_at,
        direction=direction,
        status=status,
    )


# Which direction each declared metric runs. Anything named max_* is a ceiling.
def _direction(metric: str) -> str:
    return "max" if metric.startswith("max_") else "min"


def check_performance(policy: dict, scores: dict, majority: float) -> list[Check]:
    perf = policy["performance"]
    checks = [
        _evaluate("performance.auc", "min_auc", scores["auc"], perf["min_auc"], "min"),
        _evaluate("performance.ece", "max_ece", scores["ece"], perf["max_ece"], "max"),
        _evaluate(
            "performance.accuracy_over_majority",
            "min_accuracy_over_majority",
            scores["accuracy"] - majority,
            perf["min_accuracy_over_majority"],
            "min",
        ),
    ]
    return checks


def check_fairness(policy: dict, summaries: list[dict]) -> list[Check]:
    """summaries: one dict per protected attribute, as produced by src.fairness."""
    by_attr = {s["attribute"]: s for s in summaries}
    checks = []
    for attribute, metrics in policy["fairness"].items():
        summary = by_attr.get(attribute)
        if summary is None:
            raise PolicyError(
                f"policy.yaml gates attribute {attribute!r}, but the audit produced no "
                f"summary for it. Audited: {sorted(by_attr)}"
            )
        for metric, bounds in metrics.items():
            key = _summary_key(metric)
            if key not in summary:
                # Silently skipping here would disable a gate because of a typo, which
                # is the one failure this project cannot afford.
                raise PolicyError(
                    f"policy.yaml declares {attribute}.{metric}, which maps to "
                    f"{key!r}, but the audit does not produce that. "
                    f"Available: {sorted(k for k in summary if k != 'attribute')}"
                )
            measured = summary[key]
            checks.append(
                _evaluate(
                    f"fairness.{attribute}.{metric}",
                    metric,
                    measured,
                    bounds,
                    _direction(metric),
                )
            )
    return checks


def _summary_key(metric: str) -> str:
    """Map a policy metric name onto the key src.fairness produces."""
    return {
        "max_tpr_gap": "tpr_gap",
        "max_fpr_gap": "fpr_gap",
        "max_calibration_gap": "calibration_gap",
        "max_demographic_parity_difference": "demographic_parity_difference",
        "max_equalized_odds_difference": "equalized_odds_difference",
    }.get(metric, metric)


def check_shift_canary(policy: dict, shift_scores: dict, test_gap: float,
                       shift_gap: float) -> list[Check]:
    canary = policy.get("shift_canary")
    if not canary:
        return []
    checks = [
        _evaluate("shift.auc", "min_auc", shift_scores["auc"], canary["min_auc"], "min")
    ]
    bounds = canary.get("max_tpr_gap_increase_vs_test")
    if bounds is not None:
        c = _evaluate(
            "shift.tpr_gap_increase_vs_test",
            "max_tpr_gap_increase_vs_test",
            shift_gap - test_gap,
            bounds,
            "max",
        )
        c.note = f"test gap {test_gap:.4f}, shift gap {shift_gap:.4f}"
        checks.append(c)
    return checks


def check_regression(policy: dict, current: dict, baseline: dict | None) -> list[Check]:
    """Compare against the committed baseline. No baseline means nothing to regress from."""
    rules = policy.get("regression")
    if not rules or not baseline:
        return []

    checks = []
    auc_drop = baseline["auc"] - current["auc"]
    checks.append(
        _evaluate(
            "regression.auc_drop",
            "max_auc_drop",
            auc_drop,
            {"fail": rules["max_auc_drop"]},
            "max",
        )
    )
    ece_rise = current["ece"] - baseline["ece"]
    checks.append(
        _evaluate(
            "regression.ece_increase",
            "max_ece_increase",
            ece_rise,
            {"fail": rules["max_ece_increase"]},
            "max",
        )
    )
    for key, value in current.get("gaps", {}).items():
        before = baseline.get("gaps", {}).get(key)
        if before is None:
            continue
        checks.append(
            _evaluate(
                f"regression.{key}",
                "max_gap_increase",
                value - before,
                {"fail": rules["max_gap_increase"]},
                "max",
            )
        )
    return checks


def verdict(checks: list[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    if any(c.status == WARN for c in checks):
        return WARN
    return PASS


def as_records(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def summarise(checks: list[Check]) -> str:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for c in checks:
        counts[c.status] += 1
    return (
        f"{len(checks)} checks: {counts[PASS]} pass, "
        f"{counts[WARN]} warn, {counts[FAIL]} fail"
    )
