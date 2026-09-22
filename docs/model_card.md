<!-- GENERATED FILE. Do not edit by hand.
     Produced by scripts/run_audit.py from results/audit.json.
     Editing this file is how a model card becomes fiction: change the model or
     the policy, then regenerate. -->

# Model card

Generated 2026-09-22T21:55:27+00:00 from commit `e70cd81` on Python 3.14.3.

Trained on 2015, threshold chosen on 2016, evaluated on 2018.

## What it does

Binary classification: does a person's annual income exceed $50,000? Ten features from
the American Community Survey, via the `folktables` package. Real survey responses about
real people, with recorded outcomes.

## What it is for

Nothing. This is a benchmark task used to have a real model to hold to a policy. It is
not deployed, and no decision about any person is or should be made with it.

## How it was built and validated

Gradient-boosted trees (LightGBM), with isotonic calibration fitted on the validation
year. The decision threshold is 0.345, chosen on 2016 by
maximising Youden's J, and applied unchanged to every group.

Validation is **temporal**: fitted on 2015, tuned on
2016, tested on 2018. A further split holds out four
states entirely, adding geographic shift. The test year is read once, at the end.

| split | n | AUC | avg precision | Brier | ECE | accuracy | majority baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 583,297 | 0.9002 | 0.8293 | 0.1220 | 0.0000 | 0.8117 | 0.6414 |
| test | 600,551 | 0.8881 | 0.8244 | 0.1329 | 0.0257 | 0.8021 | 0.6140 |
| shift | 43,101 | 0.8642 | 0.7356 | 0.1355 | 0.0351 | 0.7712 | 0.6980 |

The validation ECE is near zero because the calibrator was fitted on that split. It is
not a result. The meaningful numbers are the test and shift rows, where calibration error
grows as the data moves away from what the model was fitted on.

## Where it performs worse

By race (RAC1P), on the test year:

| group | n | base rate | selection rate | TPR | FPR | accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| White alone | 434,022 | 0.411 | 0.477 | 0.833 | 0.228 | 0.797 | 0.027 |
| Asian alone | 55,374 | 0.458 | 0.516 | 0.854 | 0.231 | 0.808 | 0.042 |
| Black or African American alone | 49,941 | 0.277 | 0.315 | 0.709 | 0.165 | 0.800 | 0.009 |
| Some other race alone | 40,106 | 0.188 | 0.172 | 0.542 | 0.086 | 0.844 | 0.012 |
| Two or more races | 16,928 | 0.329 | 0.367 | 0.781 | 0.164 | 0.818 | 0.028 |
| American Indian alone | 2,340 | 0.283 | 0.332 | 0.727 | 0.177 | 0.796 | 0.022 |
| American Indian and Alaska Native tribes specified | 926 | 0.194 | 0.232 | 0.694 | 0.121 | 0.843 | 0.033 |
| Native Hawaiian and Other Pacific Islander alone | 883 | 0.298 | 0.357 | 0.719 | 0.203 | 0.773 | 0.037 |

1 group(s) with fewer than 500 people were measured but are not shown or concluded from, because a rate computed on a few hundred people is mostly noise.

By sex:

| group | n | base rate | selection rate | TPR | FPR | accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male | 313,136 | 0.451 | 0.518 | 0.844 | 0.250 | 0.793 | 0.034 |
| Female | 287,415 | 0.316 | 0.360 | 0.773 | 0.170 | 0.812 | 0.016 |

Summary gaps:

| attribute | demographic parity diff | equalised odds diff | TPR gap | FPR gap | calibration gap | base rate gap |
|---|---:|---:|---:|---:|---:|---:|
| RAC1P | 0.344 | 0.312 | 0.312 | 0.145 | 0.041 | 0.270 |
| SEX | 0.158 | 0.080 | 0.071 | 0.080 | 0.034 | 0.135 |

### What that means

Among people who genuinely earn above the threshold, the model identifies **85.4%** of the *Asian alone* group and **54.2%** of the *Some other race alone* group. Accuracy is roughly flat across groups, which is exactly why accuracy is the wrong thing to look at here: a model can be equally accurate everywhere and still distribute its errors in a way that matters.

The demographic parity difference of 0.344 sits
against a **base rate gap of 0.270**: the groups genuinely differ in
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

| status | check | measured | fails at | target |
|---|---|---:|---:|---:|
| WARN | `fairness.RAC1P.max_tpr_gap` | 0.3119 | 0.350 | 0.150 |
| WARN | `fairness.RAC1P.max_fpr_gap` | 0.1447 | 0.200 | 0.100 |
| WARN | `fairness.RAC1P.max_calibration_gap` | 0.0410 | 0.080 | 0.040 |
| WARN | `fairness.RAC1P.max_demographic_parity_difference` | 0.3445 | 0.450 | 0.300 |
| WARN | `fairness.SEX.max_fpr_gap` | 0.0802 | 0.150 | 0.080 |
| WARN | `fairness.SEX.max_demographic_parity_difference` | 0.1575 | 0.250 | 0.150 |
| PASS | `performance.auc` | 0.8881 | 0.850 | 0.880 |
| PASS | `performance.ece` | 0.0257 | 0.050 | 0.030 |
| PASS | `performance.accuracy_over_majority` | 0.1881 | 0.100 | 0.150 |
| PASS | `fairness.SEX.max_tpr_gap` | 0.0710 | 0.200 | 0.080 |
| PASS | `fairness.SEX.max_calibration_gap` | 0.0344 | 0.080 | 0.040 |
| PASS | `shift.auc` | 0.8642 | 0.800 | 0.850 |
| PASS | `shift.tpr_gap_increase_vs_test` | 0.0790 | 0.150 | 0.080 |

Verdict: **WARN**. Thresholds are declared in `policy.yaml`.
