# Baselines across training seeds

Status: complete; 45/45 condition runs available.

Historical reward: -0.5 * sum(landmark nearest-agent distances) - overlapping pairs / 3; landmark spacing 0.4, agent spacing 0, loss 0, horizon 100. Each new policy is trained from scratch under its own communication schedule. Validation alone selects checkpoints. Held-out scenarios are paired across all policies (400000-400199).

All included rows are freshly measured from checkpoint evaluations. Seed 42 is rerun from scratch using the same source snapshot, training budgets and held-out scenarios as seeds 43-56. Historical rounded values are excluded.

Means give each training seed equal weight. SD and 95% percentile bootstrap intervals (20,000 resamples, fixed analysis RNG seed) describe variation across training seeds. Partial results are provisional; no significance claims are made.

| Condition | Seeds | Success mean | Seed SD | 95% seed bootstrap interval | Coverage mean | Task return mean | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 15 | 87.43% | 4.40% | 85.10% to 89.40% | 94.38% | -29.042 | 600.0 | 5.12% |
| none | 15 | 87.77% | 4.21% | 85.57% to 89.73% | 93.99% | -31.670 | 0.0 | 6.17% |
| fixed_5 | 15 | 88.47% | 3.54% | 86.77% to 90.23% | 94.31% | -29.644 | 120.0 | 5.30% |

## Individual training seeds

| Seed | Condition | Success | Coverage | Task return | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | full | 92.0% | 96.00% | -27.771 | 600 | 4.72% |
| 42 | none | 87.5% | 94.17% | -31.554 | 0 | 6.20% |
| 42 | fixed_5 | 85.5% | 93.33% | -29.891 | 120 | 4.93% |
| 43 | full | 91.5% | 96.50% | -27.206 | 600 | 4.13% |
| 43 | none | 90.5% | 96.00% | -28.035 | 0 | 4.39% |
| 43 | fixed_5 | 83.5% | 91.67% | -30.915 | 120 | 5.68% |
| 44 | full | 90.0% | 96.00% | -28.417 | 600 | 5.33% |
| 44 | none | 94.5% | 97.67% | -27.677 | 0 | 4.76% |
| 44 | fixed_5 | 87.5% | 94.67% | -28.359 | 120 | 4.99% |
| 45 | full | 85.5% | 94.33% | -28.042 | 600 | 4.33% |
| 45 | none | 88.0% | 94.00% | -31.483 | 0 | 5.99% |
| 45 | fixed_5 | 87.0% | 93.50% | -31.790 | 120 | 6.01% |
| 46 | full | 86.5% | 94.33% | -30.483 | 600 | 5.44% |
| 46 | none | 83.0% | 90.83% | -33.943 | 0 | 7.17% |
| 46 | fixed_5 | 85.5% | 92.83% | -31.264 | 120 | 5.67% |
| 47 | full | 90.5% | 95.83% | -27.661 | 600 | 4.62% |
| 47 | none | 88.5% | 93.33% | -32.242 | 0 | 6.58% |
| 47 | fixed_5 | 91.0% | 95.67% | -27.119 | 120 | 4.93% |
| 48 | full | 82.0% | 90.83% | -32.158 | 600 | 6.35% |
| 48 | none | 90.5% | 95.00% | -32.509 | 0 | 6.75% |
| 48 | fixed_5 | 87.5% | 95.00% | -31.733 | 120 | 5.84% |
| 49 | full | 83.0% | 92.50% | -27.264 | 600 | 3.84% |
| 49 | none | 91.0% | 96.00% | -30.738 | 0 | 5.43% |
| 49 | fixed_5 | 95.0% | 97.50% | -26.664 | 120 | 3.99% |
| 50 | full | 89.5% | 95.67% | -29.337 | 600 | 5.09% |
| 50 | none | 91.0% | 95.83% | -29.339 | 0 | 5.18% |
| 50 | fixed_5 | 91.0% | 94.83% | -28.831 | 120 | 4.93% |
| 51 | full | 86.5% | 94.33% | -29.837 | 600 | 5.64% |
| 51 | none | 86.0% | 93.17% | -30.223 | 0 | 5.47% |
| 51 | fixed_5 | 88.5% | 94.50% | -30.397 | 120 | 5.26% |
| 52 | full | 92.0% | 96.00% | -28.118 | 600 | 5.16% |
| 52 | none | 85.5% | 93.00% | -34.336 | 0 | 6.67% |
| 52 | fixed_5 | 83.0% | 90.83% | -31.183 | 120 | 6.06% |
| 53 | full | 76.0% | 89.17% | -32.247 | 600 | 6.65% |
| 53 | none | 77.5% | 88.00% | -38.014 | 0 | 9.14% |
| 53 | fixed_5 | 87.0% | 93.17% | -31.424 | 120 | 6.12% |
| 54 | full | 88.0% | 94.17% | -28.522 | 600 | 4.72% |
| 54 | none | 92.0% | 96.67% | -29.373 | 0 | 5.74% |
| 54 | fixed_5 | 93.5% | 97.00% | -26.723 | 120 | 4.23% |
| 55 | full | 89.0% | 95.33% | -28.280 | 600 | 4.57% |
| 55 | none | 86.5% | 94.00% | -32.777 | 0 | 7.00% |
| 55 | fixed_5 | 89.0% | 94.00% | -29.801 | 120 | 6.11% |
| 56 | full | 89.5% | 94.67% | -30.293 | 600 | 6.24% |
| 56 | none | 84.5% | 92.17% | -32.809 | 0 | 6.10% |
| 56 | fixed_5 | 92.5% | 96.17% | -28.562 | 120 | 4.83% |

Machine-readable results: `per-seed.csv`, `summary.json`; new run artifacts: `seed-N/{full,none,fixed_5}/`. `status.json` records running/completed/failed jobs. Source and historical report copies plus hashes are saved with this study.
