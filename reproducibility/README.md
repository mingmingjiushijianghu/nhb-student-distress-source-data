# Retrospective evaluation in a simulated student cohort

This directory documents a summary check for a developmental evaluation using synthetic student-cohort records. The evaluation used complete-follow-up features and retrospective threshold selection. The archived estimates describe performance on those simulated trajectories. Prospective performance requires observed data and features computed at each prediction time.

## Run the summary check

Run with Python 3 and the two archived result tables from the same analysis run:

```sh
python reproducibility/simulation_summary.py cohort_prediction.csv cohort_lead_time.csv
```

The command reads the tables supplied as arguments and prints aggregate JSON. It requires the Python standard library. The `auprc` field in the prediction archive is reported as average precision. The alert counts use the indicator for an alert strictly before the recorded confirmation. Median lead time is calculated over alerted events.

| Model comparison | AUROC | Average precision |
| --- | ---: | ---: |
| Screening scales | 0.725 | 0.054 |
| Gradient boosting on recent summaries | 0.775 | 0.083 |
| Gradient boosting with integrated features | 0.853 | 0.190 |
| Integrated features without laboratory terms | 0.876 | 0.267 |

The alert summary covers 31 of 42 simulated event labels, with a median lead time of 36 days among alerted events. Counts by trajectory are 24 of 27 persistent, 5 of 8 abrupt, 2 of 6 fluctuating, and 0 of 1 stable. The laboratory-feature comparison and the alert summaries are descriptive outputs of this run.

The companion [core analysis code](../core_code_20261118/) documents model features, input schemas and dependencies. The summary utility checks the archived outputs. Access to the simulation inputs follows the project’s data documentation.
