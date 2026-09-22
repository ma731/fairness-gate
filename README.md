# fairness-gate

A CI gate for model fairness. Declared thresholds live in `policy.yaml`, an audit run
measures the model against them, and a breach fails the build.

The model underneath is an income classifier on real US Census data, trained on one year
and tested on a later one. It is there to have something real to gate. The part that
matters is the enforcement: the model card, the EU AI Act Annex IV documentation and the
DPIA section are **generated from the audit run**, so a document cannot quietly stop being
true about the model it describes, and a threshold cannot be loosened without it showing
up in a diff someone has to approve.

> **Status: in progress.** The pipeline, the audit, the generated documents and the gate
> all work, and every number below is measured. Still to come: a mitigation with its cost
> measured, and an LLM-drafted narrative layer whose numeric claims are verified against
> the audit before they are allowed into a document. Nothing here is a product, and
> nothing here should be used to make a decision about a real person.

## The gate

```bash
python scripts/run_audit.py --gate     # full run, ~3 GB of data, exits non-zero on breach
python scripts/check_policy.py         # seconds, no data, what CI runs on every commit
```

`policy.yaml` declares two levels for every metric. `fail` breaks the build. `warn` is the
standard the project is actually aiming at. The current model sits between them on the
race gaps, which is deliberate: pinning `fail` to wherever the model happens to land today
would make the gate meaningless, and pinning it to the aspiration would mean a permanently
red build that everyone learns to ignore. Both numbers are stated so the distance between
"tolerated" and "wanted" is visible rather than quietly collapsed.

The current run is **7 pass, 6 warn, 0 fail**.

`scripts/check_policy.py` enforces three things on every commit:

1. **The committed results still satisfy `policy.yaml`**, re-evaluated rather than trusted,
   because the stored verdict was produced under whatever the policy said at the time.
2. **Every document is exactly what the results generate.** The model card, Annex IV
   document and DPIA are regenerated and compared byte for byte. Hand-edit a number in the
   model card and the build fails. This is what makes "generated, not written" a fact
   rather than a claim.
3. **The results match the code that produced them**, by content fingerprint over the
   pipeline files. Change the model and forget to re-run, and the results stop counting as
   evidence about the code they sit next to.

A scheduled workflow re-runs the whole thing monthly and opens an issue if a threshold
breaks.

---

## The task

Predict whether a person's annual income exceeds $50,000, from ten variables in the
American Community Survey, via [folktables](https://github.com/socialfoundations/folktables).

Real survey responses, real outcomes, real demographics. No synthetic data anywhere, which
matters because a fairness result computed on invented people measures nothing.

## How it is split, and why that way

| split | year | states | rows | positive rate |
|---|---|---|---:|---:|
| train | 2015 | CA, TX, NY, FL, IL | 576,014 | 0.346 |
| validation | 2016 | same | 583,297 | 0.359 |
| test | 2018 | same | 600,551 | 0.386 |
| shift | 2018 | NV, MS, WV, ME | 43,101 | 0.302 |

Training on one year and testing on a later one makes the validation **temporal**, which
is how the model would actually be used: fitted on the past, applied to the present. The
held-out states add **geographic** shift on top.

Both shifts are real and visible in that table. The positive rate climbs from 0.346 to
0.386 across the years, and the held-out states sit at 0.302. A single random split of one
year would have hidden all of it.

Every threshold and every calibration decision is made on the validation year. The test
year is read once, at the end.

## Performance

| split | AUC | avg precision | Brier | ECE | accuracy | majority baseline |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.900 | 0.829 | 0.122 | 0.000 | 0.812 | 0.641 |
| test | 0.888 | 0.824 | 0.133 | 0.026 | 0.802 | 0.614 |
| shift | 0.864 | 0.736 | 0.136 | 0.035 | 0.771 | 0.698 |

**The validation ECE of 0.000 is not a result.** The calibrator was fitted on that split,
so scoring it there measures nothing. The honest numbers are the ones after it: calibration
error grows under temporal shift and grows again under geographic shift. Average precision
falls hardest on the shifted states, which is what you would expect when the base rate
drops and the positive class gets rarer.

## Fairness audit

Test year, 600,551 people. Groups below 500 people are computed but marked unreportable.

**By race (RAC1P):**

| group | n | base rate | selection rate | TPR | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| White alone | 434,022 | 0.411 | 0.477 | 0.833 | 0.228 | 0.797 |
| Asian alone | 55,374 | 0.458 | 0.516 | 0.854 | 0.231 | 0.808 |
| Black or African American alone | 49,941 | 0.277 | 0.315 | 0.709 | 0.165 | 0.800 |
| Some other race alone | 40,106 | 0.188 | 0.172 | 0.542 | 0.086 | 0.844 |
| Two or more races | 16,928 | 0.329 | 0.367 | 0.781 | 0.164 | 0.818 |
| American Indian alone | 2,340 | 0.283 | 0.332 | 0.727 | 0.177 | 0.796 |

**Summary gaps:**

| attribute | demographic parity diff | equalised odds diff | TPR gap | FPR gap | calibration gap | base rate gap |
|---|---:|---:|---:|---:|---:|---:|
| race | 0.344 | 0.312 | 0.312 | 0.145 | 0.041 | 0.270 |
| sex | 0.158 | 0.080 | 0.071 | 0.080 | 0.034 | 0.135 |

### How to read this without lying

More than one criterion is reported on purpose. Demographic parity, equalised odds and
calibration **cannot all hold at once** when base rates differ across groups, and here they
differ a lot: the base rate gap by race is 0.270. Any single number, quoted alone, is a
choice about which unfairness to make invisible.

So the pack states which criterion the model fails and by how much, rather than picking the
flattering one. The 0.344 parity difference substantially tracks a real difference in
recorded outcomes. The 0.312 true positive rate gap does not have that excuse: among people
who actually earn above the threshold, the model finds 85.4% of the Asian group and 54.2%
of the "some other race" group. That is the number that would worry a reviewer, and it
should.

Accuracy is roughly flat across groups, and is the least informative metric here. A model
can be equally accurate everywhere and still distribute its errors in a way that matters.

## What this does not claim

- **It is not a deployable system**, and income prediction on census data is a benchmark,
  not a product. Treat every number as a measurement of this pipeline, not advice about
  anybody.
- **Race and sex categories are Census Bureau recodes**, with the coarseness and the
  politics that implies. "Some other race alone" is not a community; it is a residual.
- **Small groups are reported but not concluded from.** Anything under 500 people is
  marked unreportable and excluded from the summary gaps.
- Nothing here measures the harm of a wrong decision, because that depends on what the
  prediction would be used for, and this is used for nothing.

## Run it

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py      # ~3 GB of ACS CSVs, cached outside the repo
python scripts/run_audit.py          # regenerates every number above
```

## Layout

```
src/config.py      the experiment definition: states, years, seeds
src/data.py        ACS loading, the four splits, protected attributes kept apart
src/model.py       LightGBM, validation-year threshold, isotonic calibration
src/fairness.py    per-group metrics and the disparity summaries
scripts/           data download, audit run
results/           generated measurements
docs/              generated model card, Annex IV, DPIA
tests/             leakage and metric-arithmetic tests
```
