# Scheduler held-out evaluation: variance change from 200 to 500 episodes

Training seeds 42–56; exactly matched scenarios 400000–400499 across all groups; three agents and three landmarks; minimum pairwise radial agent-start and landmark spacing 0.4; zero loss; horizon 100; no collision termination. Training, validation, confirmation and checkpoint choices are unchanged. Initial none reuses the expanded baseline results. Adaptive normal results are shared between all three studies.

## Variance change: 200 versus 500 held-out episodes

Each seed's episode variance is a sample variance (denominators 199 and 499). This table first averages those variances equally across the 15 seeds, then reports 100 × (variance500 − variance200) / variance200. It is not a change in standard error or a mean of seed-specific percentage changes. Coverage variance uses fractions on the 0–1 scale; task-return variance uses squared return units. Individual seed changes can have either sign. More evaluation episodes improve the estimate; they do not inherently reduce outcome variance.

| Group | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| adaptive | task_return | 296.709703 | 317.526628 | +7.02% |
| adaptive | coverage | 0.023385 | 0.021430 | -8.36% |
| communication_disabled | task_return | 299.579645 | 322.040850 | +7.50% |
| communication_disabled | coverage | 0.022889 | 0.021792 | -4.79% |
| warm_none | task_return | 273.712187 | 300.812476 | +9.90% |
| warm_none | coverage | 0.022816 | 0.023085 | +1.18% |
| initial_none | task_return | 364.434267 | 378.087675 | +3.75% |
| initial_none | coverage | 0.036627 | 0.033954 | -7.30% |

## Variance across the 15 seed-level means

This is a separate quantity: the sample variance of the 15 evaluation means. It is not within-seed episode variance.

| Group | Metric | Across-seed variance: 200 | Across-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| adaptive | task_return | 1.18409213 | 0.70450992 | -40.50% |
| adaptive | coverage | 0.00010468 | 0.00008615 | -17.70% |
| communication_disabled | task_return | 1.22732041 | 0.72942463 | -40.57% |
| communication_disabled | coverage | 0.00009265 | 0.00009707 | +4.78% |
| warm_none | task_return | 1.35375010 | 0.99255230 | -26.68% |
| warm_none | coverage | 0.00016019 | 0.00007237 | -54.82% |
| initial_none | task_return | 9.12945464 | 6.19785394 | -32.11% |
| initial_none | coverage | 0.00102931 | 0.00072869 | -29.21% |

Full precision and all 60 seed/group comparisons: `results/scheduler-heldout500-20261112/within-seed-variance-200-vs-500.csv`. Policy-level summaries: `results/scheduler-heldout500-20261112/variance-summary.csv`.

## Baseline policy variance changes: 200 versus 500 episodes

The full, fixed_5 and none baselines use the same 15 training seeds (42–56), the original 200 scenarios and expanded 500 scenarios as the scheduler studies, with unchanged checkpoints, zero loss, and 0.4-unit minimum agent-start and landmark spacing. Baseline none is the same policy and evaluation data labeled initial_none above; it is repeated here to show all three baselines together.

### Average within-seed episode variance

Changes use the same definition as above: average the 15 seed-specific sample variances, then compute 100 × (variance500 − variance200) / variance200.

| Baseline | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| full | task_return | 286.760288 | 277.265212 | -3.31% |
| full | coverage | 0.027055 | 0.024173 | -10.65% |
| fixed_5 | task_return | 276.133443 | 281.577295 | +1.97% |
| fixed_5 | coverage | 0.022571 | 0.021406 | -5.16% |
| none | task_return | 364.434267 | 378.087675 | +3.75% |
| none | coverage | 0.036627 | 0.033954 | -7.30% |

### Variance across the 15 seed-level means

| Baseline | Metric | Across-seed variance: 200 | Across-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| full | task_return | 3.12904698 | 2.71715236 | -13.16% |
| full | coverage | 0.00041627 | 0.00035120 | -15.63% |
| fixed_5 | task_return | 2.11261148 | 1.49130504 | -29.41% |
| fixed_5 | coverage | 0.00022638 | 0.00012393 | -45.26% |
| none | task_return | 9.12945464 | 6.19785394 | -32.11% |
| none | coverage | 0.00102931 | 0.00072869 | -29.21% |

Full-precision baseline summaries: `results/scheduler-heldout500-20261112/baseline-variance-summary.csv`. All 45 per-seed baseline changes: `results/baseline-heldout500-20261112/within-seed-variance-change-200-vs-500.csv`. Source episode statistics: the `within-seed-episode-variation.csv` files in the original and expanded baseline study directories.

## Success variance changes: 200 versus 500 episodes

Success is a binary episode outcome (1 = final-step success, 0 = failure). For each policy seed its sample variance is n/(n−1) × p × (1−p), with n equal to 200 or 500. Variance uses the 0–1 outcome scale (squared fraction units). Baseline none is identical to initial_none in the tables above and is listed once here. All six conditions use the same training seeds 42–56 and paired held-out scenarios.

### Average within-seed success variance

Average the 15 episode variances equally, then compute 100 × (variance500 − variance200) / variance200. This measures outcome dispersion, not uncertainty in the estimated success rate.

| Policy | Mean success variance: 200 | Mean success variance: 500 | Change |
| --- | --- | --- | --- |
| adaptive | 0.087926 | 0.084296 | -4.13% |
| communication_disabled | 0.087404 | 0.085042 | -2.70% |
| warm_none | 0.099588 | 0.096194 | -3.41% |
| full | 0.102070 | 0.094197 | -7.71% |
| fixed_5 | 0.092171 | 0.086315 | -6.35% |
| none | 0.123645 | 0.117872 | -4.67% |

### Variance across the 15 seed-level success rates

This is the sample variance of the 15 per-seed success rates, rather than variance of binary episode outcomes within a seed.

| Policy | Across-seed success variance: 200 | Across-seed success variance: 500 | Change |
| --- | --- | --- | --- |
| adaptive | 0.00039952 | 0.00029764 | -25.50% |
| communication_disabled | 0.00038095 | 0.00031257 | -17.95% |
| warm_none | 0.00094595 | 0.00035829 | -62.12% |
| full | 0.00160238 | 0.00170907 | +6.66% |
| fixed_5 | 0.00101310 | 0.00045650 | -54.94% |
| none | 0.00379095 | 0.00316678 | -16.46% |

Full-precision success variance summaries: `results/scheduler-heldout500-20261112/success-variance-summary.csv`. All 90 policy/seed comparisons: `results/scheduler-heldout500-20261112/success-variance-per-seed-200-vs-500.csv`. Reproduce with `.venv/Scripts/python.exe scripts/add_success_variance_comparison.py`.
