"""Run the whole audit and write every artifact. One command, no hidden steps.

    python scripts/run_audit.py            # run, write results and docs
    python scripts/run_audit.py --gate     # ...and exit non-zero if policy fails
    python scripts/run_audit.py --set-baseline

Everything downstream reads results/audit.json. The model card, the Annex IV document and
the DPIA are generated from it, so a document cannot drift away from the model it claims
to describe: regenerating is the only way to change them.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import policy as pol
from src.config import RESULTS_DIR, TEST_YEAR, TRAIN_STATES, TRAIN_YEAR, VAL_YEAR
from src.data import PROTECTED, load_splits
from src.distributions import build as build_distributions
from src.fairness import audit
from src.fingerprint import pipeline_fingerprint
from src.model import baseline_rate, pick_threshold, predict_proba, score, train
from src.report import write_all
from src.uncertainty import build as build_uncertainty


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # No git, or not a repo. The audit is still valid, it just cannot say where from.
        return "unknown"


def run(gate: bool = False, set_baseline: bool = False) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    print("loading splits ...", flush=True)
    splits = load_splits()
    for name in ("train", "val", "test", "shift"):
        print("  " + splits[name].describe(), flush=True)

    print("training ...", flush=True)
    model = train(splits["train"], splits["val"])

    # Every choice that could touch the test year is made on the validation year.
    p_val = predict_proba(model, splits["val"])
    threshold = pick_threshold(splits["val"].y.to_numpy(), p_val)
    print(f"  threshold chosen on {VAL_YEAR}: {threshold:.4f}", flush=True)

    scores, group_tables, summaries, dists = {}, {}, {}, {}
    uncertainty: dict = {}
    for name in ("val", "test", "shift"):
        split = splits[name]
        p = predict_proba(model, split)
        pred = (p >= threshold).astype(int)
        sc = score(split, p, threshold)
        scores[name] = sc.as_row() | {"majority_baseline": baseline_rate(split)}
        if name in ("test", "shift"):
            table, summary = audit(split.y, p, pred, split.A, PROTECTED)
            group_tables[name] = table
            summaries[name] = summary.to_dict(orient="records")
            # Distribution series for the charts that need more than a summary row.
            # Only groups above the reporting floor, so nothing noisy gets drawn.
            dists[name] = {}
            if name == "test":
                uncertainty.update(
                    {attr: build_uncertainty(table, attr) for attr in PROTECTED}
                )
            for attr in PROTECTED:
                keep = set(
                    table[(table["attribute"] == attr) & table["reportable"]]["code"]
                )
                dists[name][attr] = build_distributions(
                    split.y, p, pred, split.A, attr, keep
                )
        print(f"  {name}: auc={sc.auc:.4f} ece={sc.ece:.4f} acc={sc.accuracy:.4f}",
              flush=True)

    policy = pol.load_policy()

    test_gaps = {
        f"{s['attribute']}.tpr_gap": s["tpr_gap"] for s in summaries["test"]
    } | {f"{s['attribute']}.fpr_gap": s["fpr_gap"] for s in summaries["test"]}

    current = {
        "auc": scores["test"]["auc"],
        "ece": scores["test"]["ece"],
        "gaps": test_gaps,
    }
    baseline = None
    if pol.BASELINE_PATH.exists():
        baseline = json.loads(pol.BASELINE_PATH.read_text(encoding="utf-8"))

    race_test = next(s for s in summaries["test"] if s["attribute"] == "RAC1P")
    race_shift = next(s for s in summaries["shift"] if s["attribute"] == "RAC1P")

    checks = (
        pol.check_performance(policy, scores["test"], scores["test"]["majority_baseline"])
        + pol.check_fairness(policy, summaries["test"], uncertainty)
        + pol.check_shift_canary(
            policy, scores["shift"], race_test["tpr_gap"], race_shift["tpr_gap"]
        )
        + pol.check_regression(policy, current, baseline)
    )

    result = {
        "generated_at": started.isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "pipeline_fingerprint": pipeline_fingerprint(),
        "design": {
            "train_year": TRAIN_YEAR,
            "val_year": VAL_YEAR,
            "test_year": TEST_YEAR,
            "train_states": TRAIN_STATES,
            "threshold": round(threshold, 4),
            "threshold_chosen_on": VAL_YEAR,
        },
        "splits": {k: {"n": len(v), "positive_rate": round(float(v.y.mean()), 4)}
                   for k, v in splits.items()},
        "scores": scores,
        "fairness": summaries,
        "distributions": dists,
        "uncertainty": uncertainty,
        "policy_verdict": pol.verdict(checks),
        "checks": pol.as_records(checks),
    }

    (RESULTS_DIR / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for name, table in group_tables.items():
        table.to_csv(RESULTS_DIR / f"groups_{name}.csv", index=False)

    write_all(result, group_tables)

    print()
    for c in checks:
        if c.status != pol.PASS:
            print("  " + c.line() + (f"  [{c.note}]" if c.note else ""), flush=True)
    print(f"\n{pol.summarise(checks)}  ->  {result['policy_verdict'].upper()}")
    print(f"written to {RESULTS_DIR}")

    if set_baseline:
        pol.BASELINE_PATH.write_text(
            json.dumps(current, indent=2) + "\n", encoding="utf-8"
        )
        print(f"baseline updated: {pol.BASELINE_PATH}")

    if gate and result["policy_verdict"] == pol.FAIL:
        print("\nPolicy gate FAILED. See the checks above.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero when a declared threshold is breached")
    ap.add_argument("--set-baseline", action="store_true",
                    help="record this run as the regression baseline (deliberate act)")
    args = ap.parse_args()
    return run(gate=args.gate, set_baseline=args.set_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
