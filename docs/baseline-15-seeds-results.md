# Baselines across training seeds

Status: complete; 45/45 condition runs available.

Reward: -0.5 * sum(landmark nearest-agent distances) - overlapping pairs / 3; minimum pairwise radial landmark center distance 0.4, minimum pairwise radial agent starting center distance 0.4, loss 0, horizon 100. Spacing applies to training, curriculum, validation, evaluation and replays. Agents may move closer after reset. Each policy is trained from scratch under its own communication schedule. Validation alone selects checkpoints. Held-out scenarios are paired across all policies (400000-400499); 500 evaluation episodes per trained policy.

Expanded held-out evaluation: 500 episodes per trained policy (22,500 total). The original 200 scenarios are preserved exactly, with 300 additional scenarios (400200–400499) evaluated using the same frozen checkpoints and historical environment code. No retraining or checkpoint reselection was performed. The original 200-episode study remains archived at `results/baseline-agent-spacing04-15-seeds-20261109/`; this expanded study is at `results/baseline-heldout500-20261112/`.

Means give each training seed equal weight. SD and 95% percentile bootstrap intervals (20,000 resamples, fixed analysis RNG seed) describe variation across training seeds. Partial results are provisional; no significance claims are made.

| Condition | Seeds | Success mean | Seed SD | 95% seed bootstrap interval | Coverage mean | Task return mean | Task return SD | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 15 | 89.29% | 4.13% | 87.16% to 91.19% | 95.05% | -27.687 | 1.648 | 600.0 | 4.86% |
| none | 15 | 85.97% | 5.63% | 83.19% to 88.72% | 93.16% | -30.911 | 2.490 | 0.0 | 5.83% |
| fixed_5 | 15 | 90.43% | 2.14% | 89.31% to 91.39% | 95.61% | -27.774 | 1.221 | 120.0 | 4.77% |

## Individual training seeds

| Seed | Condition | Success | Coverage | Task return | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | full | 92.2% | 96.07% | -26.854 | 600 | 4.72% |
| 42 | none | 88.0% | 93.53% | -31.026 | 0 | 5.43% |
| 42 | fixed_5 | 89.8% | 95.33% | -28.270 | 120 | 5.37% |
| 43 | full | 90.4% | 96.07% | -25.831 | 600 | 4.09% |
| 43 | none | 93.6% | 96.80% | -28.283 | 0 | 4.74% |
| 43 | fixed_5 | 91.0% | 96.47% | -26.502 | 120 | 4.02% |
| 44 | full | 93.0% | 96.93% | -27.036 | 600 | 4.89% |
| 44 | none | 84.2% | 92.33% | -31.426 | 0 | 5.90% |
| 44 | fixed_5 | 92.4% | 96.47% | -27.531 | 120 | 4.74% |
| 45 | full | 80.0% | 90.40% | -31.351 | 600 | 6.43% |
| 45 | none | 75.6% | 87.87% | -35.222 | 0 | 7.36% |
| 45 | fixed_5 | 91.4% | 96.33% | -27.382 | 120 | 4.47% |
| 46 | full | 90.0% | 95.53% | -28.451 | 600 | 5.09% |
| 46 | none | 85.8% | 93.00% | -31.622 | 0 | 6.32% |
| 46 | fixed_5 | 85.4% | 92.53% | -30.445 | 120 | 5.96% |
| 47 | full | 88.2% | 94.60% | -27.503 | 600 | 4.38% |
| 47 | none | 87.2% | 93.80% | -31.490 | 0 | 6.08% |
| 47 | fixed_5 | 91.4% | 96.00% | -28.260 | 120 | 4.87% |
| 48 | full | 91.0% | 95.93% | -26.442 | 600 | 4.18% |
| 48 | none | 89.8% | 95.20% | -29.470 | 0 | 4.97% |
| 48 | fixed_5 | 91.4% | 96.33% | -26.640 | 120 | 4.16% |
| 49 | full | 95.0% | 97.60% | -24.718 | 600 | 4.06% |
| 49 | none | 94.8% | 97.27% | -26.270 | 0 | 4.45% |
| 49 | fixed_5 | 87.6% | 94.80% | -28.162 | 120 | 4.49% |
| 50 | full | 90.4% | 95.87% | -27.318 | 600 | 4.37% |
| 50 | none | 93.2% | 96.80% | -26.654 | 0 | 4.46% |
| 50 | fixed_5 | 93.6% | 96.93% | -26.630 | 120 | 4.38% |
| 51 | full | 91.4% | 95.80% | -26.899 | 600 | 4.85% |
| 51 | none | 84.8% | 93.00% | -31.154 | 0 | 5.79% |
| 51 | fixed_5 | 91.2% | 96.53% | -26.112 | 120 | 3.86% |
| 52 | full | 91.8% | 96.20% | -27.204 | 600 | 4.90% |
| 52 | none | 78.4% | 90.20% | -33.733 | 0 | 7.11% |
| 52 | fixed_5 | 91.6% | 95.73% | -26.201 | 120 | 4.40% |
| 53 | full | 85.6% | 93.53% | -29.095 | 600 | 5.55% |
| 53 | none | 86.0% | 92.33% | -30.958 | 0 | 6.09% |
| 53 | fixed_5 | 88.0% | 94.80% | -28.869 | 120 | 4.91% |
| 54 | full | 91.8% | 95.33% | -27.736 | 600 | 4.87% |
| 54 | none | 82.4% | 92.07% | -30.849 | 0 | 5.98% |
| 54 | fixed_5 | 92.0% | 96.00% | -27.983 | 120 | 5.34% |
| 55 | full | 82.0% | 92.53% | -29.511 | 600 | 5.19% |
| 55 | none | 78.8% | 89.40% | -34.093 | 0 | 7.03% |
| 55 | fixed_5 | 90.8% | 95.20% | -28.626 | 120 | 5.52% |
| 56 | full | 86.6% | 93.33% | -29.350 | 600 | 5.30% |
| 56 | none | 87.0% | 93.80% | -31.420 | 0 | 5.76% |
| 56 | fixed_5 | 88.8% | 94.67% | -28.991 | 120 | 5.05% |

Machine-readable results: `per-seed.csv`, `summary.json`; new run artifacts: `seed-N/{full,none,fixed_5}/`. `status.json` records running/completed/failed jobs. Source and historical report copies plus hashes are saved with this study.

## Within-seed evaluation episode variation

Each row uses the 500 held-out evaluation episodes (scenario seeds 400000–400499) for one trained policy. Variance is sample variance with denominator 499; SD is its square root. These statistics describe episode variation within a seed, separately from the across-training-seed SDs above. Coverage is the final fraction of landmarks covered in each episode. Coverage SD is shown in percentage points; coverage variance uses fractions on the 0–1 scale (squared fraction units). Task-return variance is in squared return units.

| Seed | Policy | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |
| --- | --- | --- | --- | --- | --- |
| 42 | full | 237.065 | 15.397 | 0.021830 | 14.77 |
| 42 | none | 393.689 | 19.842 | 0.037449 | 19.35 |
| 42 | fixed_5 | 304.178 | 17.441 | 0.022757 | 15.09 |
| 43 | full | 224.175 | 14.972 | 0.016486 | 12.84 |
| 43 | none | 309.568 | 17.595 | 0.017678 | 13.30 |
| 43 | fixed_5 | 210.530 | 14.510 | 0.014113 | 11.88 |
| 44 | full | 263.119 | 16.221 | 0.014644 | 12.10 |
| 44 | none | 355.212 | 18.847 | 0.037531 | 19.37 |
| 44 | fixed_5 | 269.904 | 16.429 | 0.017676 | 13.30 |
| 45 | full | 380.010 | 19.494 | 0.045096 | 21.24 |
| 45 | none | 497.105 | 22.296 | 0.057393 | 23.96 |
| 45 | fixed_5 | 260.564 | 16.142 | 0.017134 | 13.09 |
| 46 | full | 295.824 | 17.200 | 0.020936 | 14.47 |
| 46 | none | 498.607 | 22.330 | 0.035838 | 18.93 |
| 46 | fixed_5 | 399.354 | 19.984 | 0.038502 | 19.62 |
| 47 | full | 247.752 | 15.740 | 0.026693 | 16.34 |
| 47 | none | 380.340 | 19.502 | 0.031998 | 17.89 |
| 47 | fixed_5 | 270.739 | 16.454 | 0.020218 | 14.22 |
| 48 | full | 240.862 | 15.520 | 0.019496 | 13.96 |
| 48 | none | 338.677 | 18.403 | 0.024857 | 15.77 |
| 48 | fixed_5 | 262.775 | 16.210 | 0.016244 | 12.75 |
| 49 | full | 203.830 | 14.277 | 0.013228 | 11.50 |
| 49 | none | 253.585 | 15.924 | 0.017287 | 13.15 |
| 49 | fixed_5 | 260.327 | 16.135 | 0.022675 | 15.06 |
| 50 | full | 287.685 | 16.961 | 0.018328 | 13.54 |
| 50 | none | 289.004 | 17.000 | 0.016342 | 12.78 |
| 50 | fixed_5 | 243.340 | 15.599 | 0.016426 | 12.82 |
| 51 | full | 288.861 | 16.996 | 0.022503 | 15.00 |
| 51 | none | 352.627 | 18.778 | 0.032276 | 17.97 |
| 51 | fixed_5 | 217.856 | 14.760 | 0.013937 | 11.81 |
| 52 | full | 275.941 | 16.611 | 0.020152 | 14.20 |
| 52 | none | 426.932 | 20.662 | 0.041368 | 20.34 |
| 52 | fixed_5 | 267.059 | 16.342 | 0.024005 | 15.49 |
| 53 | full | 310.913 | 17.633 | 0.029433 | 17.16 |
| 53 | none | 390.641 | 19.765 | 0.042875 | 20.71 |
| 53 | fixed_5 | 316.927 | 17.802 | 0.023565 | 15.35 |
| 54 | full | 286.731 | 16.933 | 0.028546 | 16.90 |
| 54 | none | 343.546 | 18.535 | 0.035778 | 18.92 |
| 54 | fixed_5 | 320.849 | 17.912 | 0.021554 | 14.68 |
| 55 | full | 304.178 | 17.441 | 0.030041 | 17.33 |
| 55 | none | 443.214 | 21.053 | 0.049975 | 22.36 |
| 55 | fixed_5 | 304.432 | 17.448 | 0.025747 | 16.05 |
| 56 | full | 312.033 | 17.664 | 0.035181 | 18.76 |
| 56 | none | 398.567 | 19.964 | 0.030662 | 17.51 |
| 56 | fixed_5 | 314.826 | 17.743 | 0.026542 | 16.29 |

Full-precision episode statistics: `results/baseline-heldout500-20261112/within-seed-episode-variation.csv`. Reproduce with `.venv/Scripts/python.exe scripts/baseline_episode_variance.py --root results/baseline-heldout500-20261112 --episodes 500`.
