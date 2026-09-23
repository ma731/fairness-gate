<!-- GENERATED FILE. Do not edit by hand.
     Produced by scripts/run_audit.py from results/audit.json.
     Editing this file is how a model card becomes fiction: change the model or
     the policy, then regenerate. -->

# Technical documentation (EU AI Act, Annex IV)

Generated 2026-09-23T13:29:44+00:00 from commit `21b8c3d` on Python 3.14.3.

Trained on 2015, threshold chosen on 2016, evaluated on 2018.

> **Scope note.** This system is a benchmark exercise and is not placed on the market, so
> Annex IV does not legally apply to it. The document is written to the Annex IV structure
> because that structure is the one a real high-risk system would have to answer, and
> because writing it exposes which questions a project cannot yet answer.

## 1. General description of the AI system

An income classifier over US Census microdata. Intended purpose: none beyond
demonstrating a policy-gated audit pipeline. No deployment, no users, no decisions.

Provider: individual project. Version: commit `21b8c3d`.

## 2. Elements of the system and its development

**Data.** American Community Survey public use microdata, obtained through the
`folktables` package. Training 2015, validation
2016, testing 2018. Training states:
CA, TX, NY, FL, IL. A held-out geographic split covers four further states.

| split | rows | positive rate |
|---|---:|---:|
| train | 576,014 | 0.346 |
| val | 583,297 | 0.359 |
| test | 600,551 | 0.386 |
| shift | 43,101 | 0.302 |

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
different rates across racial groups (TPR gap 0.312). Were this system
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

A mitigation was built, measured, and **not adopted**. Recording that with its numbers,
rather than asserting it would be costly, is the point of this section.

**What was tried.** One decision threshold per group, chosen on the validation year to
equalise recall, then applied unchanged to the test year.

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| One threshold for everyone | 0.3119 | 0.1447 | 0.8021 |
| One threshold per group | 0.1143 | 0.1956 | 0.7979 |

**Why it was rejected, in order of weight.**

1. It requires the protected attribute at the moment of prediction. The system must read
   a person's race to decide which threshold applies to them. In the settings anyone
   would actually want this for, that is either unlawful or is itself the harm.
2. It moves the unfairness rather than removing it. The false positive gap rises from 0.1447 to 0.1956. With unequal base rates, equalising one error rate across groups necessarily unequalises another, which is the impossibility result rather than an implementation fault.
3. The accuracy cost is small, 0.0042, and that is precisely
   why the first two reasons have to be written down. By the number most projects
   report, this looks nearly free.

**What was also tried and does not work.** Sweeping the single global threshold across its whole range. The lowest recall gap any single cut achieves is 0.0342, at a threshold of 0.05 and an accuracy of 0.6022: it equalises groups by selecting almost everyone. That is not a mitigation, it is abandoning the model.

## 6. Validation and testing

Metrics, splits and per-group results are in the model card, generated from the same run
as this document. The policy gate result for this run is **WARN**.

## 7. Post-market monitoring

Not applicable: nothing is on any market. The equivalent implemented here is a scheduled
re-audit that re-runs the full pipeline and raises an issue when a declared threshold is
breached.
