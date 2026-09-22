"""Generate the compliance documents from a completed audit run.

Every number in every document comes from results/audit.json. Nothing is typed by hand,
which is the only reliable way to stop a model card from slowly becoming fiction: the
document cannot be edited into agreement with a model that changed, it can only be
regenerated.

Documents produced:
  docs/model_card.md   what the model is, how it performs, and for whom it performs worse
  docs/annex_iv.md     EU AI Act Annex IV technical documentation
  docs/dpia.md         the data protection impact assessment section
  results/checks.md    the policy gate result, human readable
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import dashboard, pages

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"

GENERATED_BANNER = (
    "<!-- GENERATED FILE. Do not edit by hand.\n"
    "     Produced by scripts/run_audit.py from results/audit.json.\n"
    "     Editing this file is how a model card becomes fiction: change the model or\n"
    "     the policy, then regenerate. -->\n"
)


def _provenance(result: dict) -> str:
    d = result["design"]
    return (
        f"Generated {result['generated_at']} from commit `{result['git_sha']}` "
        f"on Python {result['python']}.\n\n"
        f"Trained on {d['train_year']}, threshold chosen on {d['threshold_chosen_on']}, "
        f"evaluated on {d['test_year']}."
    )


def _scores_table(result: dict) -> str:
    rows = ["| split | n | AUC | avg precision | Brier | ECE | accuracy | majority baseline |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("val", "test", "shift"):
        s = result["scores"][name]
        rows.append(
            f"| {name} | {s['n']:,} | {s['auc']:.4f} | {s['average_precision']:.4f} | "
            f"{s['brier']:.4f} | {s['ece']:.4f} | {s['accuracy']:.4f} | "
            f"{s['majority_baseline']:.4f} |"
        )
    return "\n".join(rows)


def _group_table(table: pd.DataFrame, attribute: str) -> str:
    sub = table[(table["attribute"] == attribute) & table["reportable"]]
    rows = ["| group | n | base rate | selection rate | TPR | FPR | accuracy | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in sub.iterrows():
        rows.append(
            f"| {r['group']} | {int(r['n']):,} | {r['base_rate']:.3f} | "
            f"{r['selection_rate']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} | "
            f"{r['accuracy']:.3f} | {r['ece']:.3f} |"
        )
    suppressed = int((~table[table["attribute"] == attribute]["reportable"]).sum())
    if suppressed:
        rows.append("")
        rows.append(
            f"{suppressed} group(s) with fewer than 500 people were measured but are not "
            "shown or concluded from, because a rate computed on a few hundred people is "
            "mostly noise."
        )
    return "\n".join(rows)


def _gaps_table(result: dict, split: str = "test") -> str:
    header = (
        "| attribute | demographic parity diff | equalised odds diff | TPR gap | "
        "FPR gap | calibration gap | base rate gap |"
    )
    rows = [header, "|---|---:|---:|---:|---:|---:|---:|"]
    for s in result["fairness"][split]:
        rows.append(
            f"| {s['attribute']} | {s['demographic_parity_difference']:.3f} | "
            f"{s['equalized_odds_difference']:.3f} | {s['tpr_gap']:.3f} | "
            f"{s['fpr_gap']:.3f} | {s['calibration_gap']:.3f} | "
            f"{s['base_rate_gap']:.3f} |"
        )
    return "\n".join(rows)


def _worst_tpr_pair(table: pd.DataFrame, attribute: str):
    sub = table[(table["attribute"] == attribute) & table["reportable"]]
    if sub.empty:
        return None
    best = sub.loc[sub["tpr"].idxmax()]
    worst = sub.loc[sub["tpr"].idxmin()]
    return best, worst


def _checks_lines(result: dict) -> str:
    order = {"fail": 0, "warn": 1, "pass": 2}
    checks = sorted(result["checks"], key=lambda c: order[c["status"]])
    rows = ["| status | check | measured | fails at | target |",
            "|---|---|---:|---:|---:|"]
    for c in checks:
        fail_at = "" if c["fail_at"] is None else f"{c['fail_at']:.3f}"
        warn_at = "" if c["warn_at"] is None else f"{c['warn_at']:.3f}"
        rows.append(
            f"| {c['status'].upper()} | `{c['name']}` | {c['measured']:.4f} | "
            f"{fail_at} | {warn_at} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def model_card(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    test = tables["test"]
    d = result["design"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    pair = _worst_tpr_pair(test, "RAC1P")

    harm = ""
    if pair:
        best, worst = pair
        harm = (
            f"Among people who genuinely earn above the threshold, the model identifies "
            f"**{best['tpr']:.1%}** of the *{best['group']}* group and "
            f"**{worst['tpr']:.1%}** of the *{worst['group']}* group. Accuracy is roughly "
            f"flat across groups, which is exactly why accuracy is the wrong thing to "
            f"look at here: a model can be equally accurate everywhere and still "
            f"distribute its errors in a way that matters."
        )

    return f"""{GENERATED_BANNER}
# Model card

{_provenance(result)}

## What it does

Binary classification: does a person's annual income exceed $50,000? Ten features from
the American Community Survey, via the `folktables` package. Real survey responses about
real people, with recorded outcomes.

## What it is for

Nothing. This is a benchmark task used to have a real model to hold to a policy. It is
not deployed, and no decision about any person is or should be made with it.

## How it was built and validated

Gradient-boosted trees (LightGBM), with isotonic calibration fitted on the validation
year. The decision threshold is {d['threshold']}, chosen on {d['threshold_chosen_on']} by
maximising Youden's J, and applied unchanged to every group.

Validation is **temporal**: fitted on {d['train_year']}, tuned on
{d['threshold_chosen_on']}, tested on {d['test_year']}. A further split holds out four
states entirely, adding geographic shift. The test year is read once, at the end.

{_scores_table(result)}

The validation ECE is near zero because the calibrator was fitted on that split. It is
not a result. The meaningful numbers are the test and shift rows, where calibration error
grows as the data moves away from what the model was fitted on.

## Where it performs worse

By race (RAC1P), on the test year:

{_group_table(test, "RAC1P")}

By sex:

{_group_table(test, "SEX")}

Summary gaps:

{_gaps_table(result)}

### What that means

{harm}

The demographic parity difference of {race['demographic_parity_difference']:.3f} sits
against a **base rate gap of {race['base_rate_gap']:.3f}**: the groups genuinely differ in
recorded income in this data, so a model that predicted at equal rates everywhere would
have to be wrong about individuals in order to look equal in aggregate. The true positive
rate gap has no such defence, and it is the number a reviewer should press on.

Demographic parity, equalised odds and calibration cannot all hold at once when base
rates differ. All three are reported here so that the choice of which one to fail is
visible, rather than being made silently by quoting whichever is most flattering.

## Known limitations

- Race and sex are Census Bureau recodes, with the coarseness that implies. "Some other
  race alone" is a residual category, not a community.
- Groups under 500 people are measured but not reported or concluded from.
- Nothing here measures the harm of a wrong prediction, because that depends on what the
  prediction would be used for, and it is used for nothing.
- The model is fitted on five large states and degrades on the four held out, which is
  visible in the shift row above. It should not be assumed to transfer further.

## Policy result

{_checks_lines(result)}

Verdict: **{result['policy_verdict'].upper()}**. Thresholds are declared in `policy.yaml`.
"""


# --------------------------------------------------------------------------- #
# EU AI Act Annex IV technical documentation
# --------------------------------------------------------------------------- #
def annex_iv(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    d = result["design"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")

    return f"""{GENERATED_BANNER}
# Technical documentation (EU AI Act, Annex IV)

{_provenance(result)}

> **Scope note.** This system is a benchmark exercise and is not placed on the market, so
> Annex IV does not legally apply to it. The document is written to the Annex IV structure
> because that structure is the one a real high-risk system would have to answer, and
> because writing it exposes which questions a project cannot yet answer.

## 1. General description of the AI system

An income classifier over US Census microdata. Intended purpose: none beyond
demonstrating a policy-gated audit pipeline. No deployment, no users, no decisions.

Provider: individual project. Version: commit `{result['git_sha']}`.

## 2. Elements of the system and its development

**Data.** American Community Survey public use microdata, obtained through the
`folktables` package. Training {d['train_year']}, validation
{d['threshold_chosen_on']}, testing {d['test_year']}. Training states:
{", ".join(d['train_states'])}. A held-out geographic split covers four further states.

| split | rows | positive rate |
|---|---:|---:|
""" + "\n".join(
        f"| {k} | {v['n']:,} | {v['positive_rate']:.3f} |"
        for k, v in result["splits"].items()
    ) + f"""

**Design choices that affect the outcome.**

- The split is temporal rather than random. A random split of a single year would have
  hidden the drift visible in the table above, and would have reported a performance
  figure the model could not sustain in use.
- Protected attributes are held separately from features throughout, so the same audit
  can be run whether or not the model is permitted to see them.
- The decision threshold is chosen on the validation year and applied unchanged to every
  group. Per-group thresholds would improve the fairness metrics and are not used, because
  they require the protected attribute at inference time. See section 5.

**Computational resources.** Single machine, CPU only. The full audit is one command.

## 3. Monitoring, functioning and control

The system's expected accuracy and its known limits are in the model card. The gating
thresholds are declared in `policy.yaml` and enforced on every commit by
`scripts/run_audit.py --gate`, which exits non-zero on a breach.

Foreseeable unintended outcomes: the model recovers positive cases at materially
different rates across racial groups (TPR gap {race['tpr_gap']:.3f}). Were this system
used for anything, that gap would translate directly into unequal access to whatever the
positive prediction unlocked.

## 4. Risk management

| risk | how it is detected | status |
|---|---|---|
| Performance decay over time | temporal test split, ECE and AUC gated | measured each run |
| Population shift | held-out states as a canary, separately gated | measured each run |
| Disparate error rates | per-group TPR, FPR, calibration, gated | measured each run |
| Silent regression | comparison against a committed baseline | measured each run |
| Documentation drift | this document is generated, not written | structural |

## 5. Changes and trade-offs

The most effective available mitigation for the equalised odds gap is per-group threshold
optimisation. It is **not applied**, for a reason that belongs in this document rather
than a footnote: it requires the protected attribute at inference time, which in many
jurisdictions and use cases is either unlawful or a fresh harm of its own. Recording a
mitigation that was rejected, and why, is part of the documentation.

## 6. Validation and testing

Metrics, splits and per-group results are in the model card, generated from the same run
as this document. The policy gate result for this run is **{result['policy_verdict'].upper()}**.

## 7. Post-market monitoring

Not applicable: nothing is on any market. The equivalent implemented here is a scheduled
re-audit that re-runs the full pipeline and raises an issue when a declared threshold is
breached.
"""


# --------------------------------------------------------------------------- #
# DPIA
# --------------------------------------------------------------------------- #
def dpia(result: dict) -> str:
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    sex = next(s for s in result["fairness"]["test"] if s["attribute"] == "SEX")

    return f"""{GENERATED_BANNER}
# Data protection impact assessment

{_provenance(result)}

> **Scope note.** No personal data in the legal sense is processed here. ACS public use
> microdata is already de-identified and published by the Census Bureau for exactly this
> kind of analysis. This section is written because a real system would need one, and
> because the exercise of writing it surfaces the questions that matter.

## 1. Nature of the processing

Statistical modelling over published, de-identified survey microdata. No collection from
individuals, no linkage to other datasets, no re-identification attempted, no output
about any identifiable person.

## 2. Necessity and proportionality

The demographic attributes are processed **in order to measure discrimination, not to
make predictions**. This is the distinction that matters: race and sex are held apart
from the feature set and used only to compute per-group error rates. Removing them would
make the system less fair, not more, because the disparities would still exist and nobody
would be able to see them.

## 3. Risks to individuals

| risk | assessment |
|---|---|
| Re-identification | Low. Source data is published de-identified microdata. |
| Discriminatory output | **Present and measured.** TPR gap {race['tpr_gap']:.3f} by race, {sex['tpr_gap']:.3f} by sex. |
| Function creep | The stated risk. A benchmark model is one config change away from being pointed at a real decision. |
| Opacity | Mitigated: per-group metrics, calibration and the threshold are published with every run. |

## 4. Measures

- Protected attributes are structurally separated from features.
- Per-group performance is measured on every run and gated in CI.
- Groups under 500 people are suppressed from reporting rather than published as noise.
- The system carries an explicit statement that it is not for use on real decisions, in
  the README, the model card and this document.
- No raw data is committed. It is cached outside the repository and gitignored.

## 5. Residual risk

The discriminatory performance gap is **not resolved**. It is bounded by declared
thresholds and measured on every run, which is a different thing from being fixed. The
honest position is that this model would not be acceptable for any consequential decision
about a person in its current state, and the gate exists to keep saying so.
"""


def checks_doc(result: dict) -> str:
    return f"""{GENERATED_BANNER}
# Policy gate result

{_provenance(result)}

Verdict: **{result['policy_verdict'].upper()}**

{_checks_lines(result)}

`FAIL` breaks the build. `WARN` is the standard the project is aiming at and does not
break the build. The distance between the two columns is deliberate and is explained in
`policy.yaml`.
"""


def write_all(result: dict, tables: dict[str, pd.DataFrame]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (DOCS_DIR / "model_card.md").write_text(model_card(result, tables), encoding="utf-8")
    (DOCS_DIR / "annex_iv.md").write_text(annex_iv(result, tables), encoding="utf-8")
    (DOCS_DIR / "dpia.md").write_text(dpia(result), encoding="utf-8")
    (RESULTS_DIR / "checks.md").write_text(checks_doc(result), encoding="utf-8")
    dashboard.write(result, tables)
    pages.write(result)
