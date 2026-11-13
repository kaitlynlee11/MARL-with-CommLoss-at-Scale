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
