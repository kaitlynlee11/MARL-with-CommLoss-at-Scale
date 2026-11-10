# Baselines across training seeds

Status: complete; 45/45 condition runs available.

Reward: -0.5 * sum(landmark nearest-agent distances) - overlapping pairs / 3; minimum pairwise radial landmark center distance 0.4, minimum pairwise radial agent starting center distance 0.4, loss 0, horizon 100. Spacing applies to training, curriculum, validation, evaluation and replays. Agents may move closer after reset. Each policy is trained from scratch under its own communication schedule. Validation alone selects checkpoints. Held-out scenarios are paired across all policies (400000-400199).

All included rows are freshly measured from checkpoint evaluations. Seed 42 is rerun from scratch using the same source snapshot, training budgets and held-out scenarios as seeds 43-56. Historical rounded values are excluded.

Means give each training seed equal weight. SD and 95% percentile bootstrap intervals (20,000 resamples, fixed analysis RNG seed) describe variation across training seeds. Partial results are provisional; no significance claims are made.

| Condition | Seeds | Success mean | Seed SD | 95% seed bootstrap interval | Coverage mean | Task return mean | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 15 | 88.33% | 4.00% | 86.27% to 90.20% | 94.50% | -28.055 | 600.0 | 4.77% |
| none | 15 | 85.13% | 6.16% | 82.03% to 88.10% | 92.64% | -31.141 | 0.0 | 5.72% |
| fixed_5 | 15 | 89.67% | 3.18% | 88.00% to 91.10% | 95.29% | -27.893 | 120.0 | 4.71% |

## Individual training seeds

| Seed | Condition | Success | Coverage | Task return | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | full | 92.0% | 96.00% | -27.663 | 600 | 4.86% |
| 42 | none | 86.5% | 92.50% | -31.136 | 0 | 5.22% |
| 42 | fixed_5 | 90.0% | 95.67% | -27.637 | 120 | 4.96% |
| 43 | full | 88.5% | 95.00% | -26.638 | 600 | 4.25% |
| 43 | none | 93.5% | 97.00% | -28.512 | 0 | 4.53% |
| 43 | fixed_5 | 91.0% | 96.33% | -26.482 | 120 | 3.90% |
| 44 | full | 92.5% | 97.00% | -26.832 | 600 | 4.61% |
| 44 | none | 84.0% | 91.67% | -32.060 | 0 | 6.11% |
| 44 | fixed_5 | 91.0% | 95.83% | -27.832 | 120 | 4.66% |
| 45 | full | 78.5% | 89.17% | -32.599 | 600 | 6.74% |
| 45 | none | 73.5% | 86.83% | -36.328 | 0 | 7.87% |
| 45 | fixed_5 | 90.5% | 95.67% | -28.257 | 120 | 4.71% |
| 46 | full | 90.0% | 95.33% | -29.360 | 600 | 5.44% |
| 46 | none | 86.0% | 93.50% | -32.356 | 0 | 5.99% |
| 46 | fixed_5 | 85.0% | 92.33% | -30.563 | 120 | 5.83% |
| 47 | full | 85.5% | 93.83% | -28.393 | 600 | 4.26% |
| 47 | none | 87.0% | 93.67% | -30.735 | 0 | 5.69% |
| 47 | fixed_5 | 91.5% | 95.67% | -28.045 | 120 | 4.79% |
| 48 | full | 88.5% | 94.67% | -26.765 | 600 | 4.24% |
| 48 | none | 90.0% | 95.33% | -29.372 | 0 | 4.44% |
| 48 | fixed_5 | 93.5% | 97.67% | -25.932 | 120 | 3.81% |
| 49 | full | 93.5% | 97.00% | -24.907 | 600 | 3.91% |
| 49 | none | 93.0% | 96.67% | -25.506 | 0 | 4.18% |
| 49 | fixed_5 | 82.5% | 92.83% | -28.578 | 120 | 4.73% |
| 50 | full | 89.5% | 95.33% | -27.520 | 600 | 4.55% |
| 50 | none | 92.5% | 96.33% | -26.495 | 0 | 4.45% |
| 50 | fixed_5 | 92.5% | 96.33% | -26.916 | 120 | 4.13% |
| 51 | full | 89.5% | 95.17% | -27.110 | 600 | 4.29% |
| 51 | none | 84.5% | 92.83% | -30.605 | 0 | 5.51% |
| 51 | fixed_5 | 89.5% | 96.00% | -26.069 | 120 | 3.87% |
| 52 | full | 91.0% | 95.83% | -27.888 | 600 | 4.53% |
| 52 | none | 78.5% | 89.67% | -34.731 | 0 | 7.29% |
| 52 | fixed_5 | 90.5% | 95.00% | -26.196 | 120 | 4.36% |
| 53 | full | 85.5% | 92.67% | -29.171 | 600 | 5.30% |
| 53 | none | 88.5% | 93.67% | -30.482 | 0 | 5.75% |
| 53 | fixed_5 | 85.5% | 93.83% | -29.063 | 120 | 4.77% |
| 54 | full | 91.5% | 95.17% | -27.436 | 600 | 4.60% |
| 54 | none | 81.0% | 91.33% | -30.961 | 0 | 5.78% |
| 54 | fixed_5 | 94.0% | 97.17% | -27.361 | 120 | 5.06% |
| 55 | full | 83.5% | 93.00% | -28.888 | 600 | 4.74% |
| 55 | none | 75.0% | 86.33% | -35.645 | 0 | 7.34% |
| 55 | fixed_5 | 88.5% | 93.83% | -29.344 | 120 | 5.79% |
| 56 | full | 85.5% | 92.33% | -29.663 | 600 | 5.23% |
| 56 | none | 83.5% | 92.33% | -32.196 | 0 | 5.70% |
| 56 | fixed_5 | 89.5% | 95.17% | -30.129 | 120 | 5.32% |

Machine-readable results: `per-seed.csv`, `summary.json`; new run artifacts: `seed-N/{full,none,fixed_5}/`. `status.json` records running/completed/failed jobs. Source and historical report copies plus hashes are saved with this study.
