# Responsible AI evidence pack

An income classifier built on real US Census data, and the compliance evidence a
Responsible AI review would actually ask for: group fairness metrics, a mitigation with
its cost stated, a model card, EU AI Act Annex IV technical documentation and a DPIA
section.

The point is not the model. The point is that **the governance artifacts are generated
from the audit run and enforced in CI**, so they cannot quietly stop being true.

> **Status: in progress.** The data, splits, model and fairness audit work and the numbers
> below are real. The generated compliance artifacts and the CI policy gate are being
> built. Nothing here is a product, and nothing here should be used to make a decision
> about a real person.

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
