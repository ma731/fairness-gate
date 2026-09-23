<!-- GENERATED FILE. Do not edit by hand.
     Produced by scripts/run_audit.py from results/audit.json.
     Editing this file is how a model card becomes fiction: change the model or
     the policy, then regenerate. -->

# Data protection impact assessment

Generated 2026-09-23T09:25:51+00:00 from commit `45f861c` on Python 3.12.14.

Trained on 2015, threshold chosen on 2016, evaluated on 2018.

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
| Discriminatory output | **Present and measured.** TPR gap 0.312 by race, 0.071 by sex. |
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
