<!-- GENERATED FILE. Do not edit by hand.
     Produced by scripts/run_audit.py from results/audit.json.
     Editing this file is how a model card becomes fiction: change the model or
     the policy, then regenerate. -->

# Policy gate result

Generated 2026-09-23T09:54:27+00:00 from commit `e3b37e2` on Python 3.14.3.

Trained on 2015, threshold chosen on 2016, evaluated on 2018.

Verdict: **WARN**

| status | check | measured | fails at | target |
|---|---|---:|---:|---:|
| WARN | `fairness.RAC1P.max_tpr_gap` | 0.3119 | 0.350 | 0.150 |
| WARN | `fairness.RAC1P.max_fpr_gap` | 0.1447 | 0.200 | 0.100 |
| WARN | `fairness.RAC1P.max_calibration_gap` | 0.0410 | 0.080 | 0.040 |
| WARN | `fairness.RAC1P.max_demographic_parity_difference` | 0.3445 | 0.450 | 0.300 |
| WARN | `fairness.RACExSEX.max_tpr_gap` | 0.3285 | 0.500 | 0.250 |
| WARN | `fairness.RACExSEX.max_fpr_gap` | 0.2209 | 0.280 | 0.140 |
| WARN | `fairness.RACExSEX.max_demographic_parity_difference` | 0.4405 | 0.550 | 0.380 |
| WARN | `fairness.SEX.max_fpr_gap` | 0.0802 | 0.150 | 0.080 |
| WARN | `fairness.SEX.max_demographic_parity_difference` | 0.1575 | 0.250 | 0.150 |
| PASS | `performance.auc` | 0.8881 | 0.850 | 0.880 |
| PASS | `performance.ece` | 0.0257 | 0.050 | 0.030 |
| PASS | `performance.accuracy_over_majority` | 0.1881 | 0.100 | 0.150 |
| PASS | `fairness.SEX.max_tpr_gap` | 0.0710 | 0.200 | 0.080 |
| PASS | `fairness.SEX.max_calibration_gap` | 0.0344 | 0.080 | 0.040 |
| PASS | `shift.auc` | 0.8642 | 0.800 | 0.850 |
| PASS | `shift.tpr_gap_increase_vs_test` | 0.0790 | 0.150 | 0.080 |

`FAIL` breaks the build. `WARN` is the standard the project is aiming at and does not
break the build. The distance between the two columns is deliberate and is explained in
`policy.yaml`.
