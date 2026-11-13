# Adaptive scheduler: 15 seeds, zero communication loss

Status: complete; 15/15 seeds evaluated.

Targets: final-step success strictly above 90% AND mean attempted directed packets strictly below 120 per 100-step episode. Both targets use point estimates. Traffic is an episode average, not a hard cap on individual episodes.

Three agents, three static landmarks, zero communication loss, pairwise radial agent-start and landmark separation >=0.4, sensing radius 0.6 and coverage radius 0.1. Episodes run all 100 steps; collisions are recorded without terminating episodes. Collision step rate is the fraction of steps with any overlapping agent pair. This matches the corrected baseline workflow.

Each seed warm-starts its own full-communication baseline actor (8,960,000 baseline training steps); movement and transmission then learn jointly with a fresh critic and optimizer. Additional adaptive steps are reported separately. Shared base task-return scoring matches the baselines (legacy-distance-v0); adaptive shaping rewards differ. This is not a comparison at equal training compute.

Validation uses 200 batched scenarios (seed 100000). Passing validation triggers separate 200-episode original-environment confirmation (500000-500199). Training stops on confirmed targets or its step ceiling. Final frozen-policy evaluation uses the baseline-matched 500 scenarios (400000-400499); held-out results never select or tune checkpoints. A confirmation pass does not guarantee a held-out pass.

| Completed seeds | Success mean | Success seed SD | Mean packets/episode | Collision step rate | Both targets on held-out |
| --- | --- | --- | --- | --- | --- |
| 15 | 90.69% | 1.73% | 0.21 | 5.06% | 10/15 |

Aggregate point-estimate targets met: **True** (final).

| Metric | Mean | Seed SD | 95% seed bootstrap interval |
| --- | --- | --- | --- |
| success_rate | 0.906933 | 0.01725219102717379 | [0.8982666666666667, 0.9150666666666668] |
| final_coverage | 0.956978 | 0.009281716681210687 | [0.9523111111111111, 0.9614222222222222] |
| task_return | -28.671164 | 0.8393508948881855 | [-29.093850959056027, -28.27919211207355] |
| attempted_links | 0.209867 | 0.812810104924063 | [0.0, 0.6296] |
| collision_step_rate | 0.050596 | 0.005024404442319509 | [0.04825066666666668, 0.053128033333333345] |

## Individual training seeds

| Seed | Held-out success (95% Wilson) | Coverage | Packets/episode | Collision step rate | Adaptive steps | Confirmation pass | Held-out pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 90.2% (87.3%-92.5%) | 95.8% | 0.00 | 4.73% | 12,320,000 | True | True |
| 43 | 88.8% (85.7%-91.3%) | 94.9% | 0.00 | 4.45% | 10,720,000 | True | False |
| 44 | 92.4% (89.7%-94.4%) | 96.7% | 0.00 | 4.80% | 14,720,000 | True | True |
| 45 | 90.6% (87.7%-92.9%) | 95.4% | 3.15 | 5.24% | 17,120,000 | True | True |
| 46 | 89.8% (86.8%-92.2%) | 95.4% | 0.00 | 4.85% | 12,000,000 | True | False |
| 47 | 90.6% (87.7%-92.9%) | 95.8% | 0.00 | 4.43% | 12,000,000 | True | True |
| 48 | 93.6% (91.1%-95.4%) | 97.3% | 0.00 | 4.71% | 14,880,000 | True | True |
| 49 | 91.6% (88.8%-93.7%) | 96.3% | 0.00 | 4.93% | 11,680,000 | True | True |
| 50 | 89.4% (86.4%-91.8%) | 95.2% | 0.00 | 4.88% | 11,040,000 | True | False |
| 51 | 89.4% (86.4%-91.8%) | 94.8% | 0.00 | 5.77% | 13,120,000 | True | False |
| 52 | 91.2% (88.4%-93.4%) | 95.3% | 0.00 | 5.93% | 14,080,000 | True | True |
| 53 | 91.6% (88.8%-93.7%) | 95.9% | 0.00 | 5.53% | 12,320,000 | True | True |
| 54 | 91.6% (88.8%-93.7%) | 96.0% | 0.00 | 5.21% | 17,440,000 | True | True |
| 55 | 92.8% (90.2%-94.8%) | 96.9% | 0.00 | 4.58% | 12,320,000 | True | True |
| 56 | 86.8% (83.6%-89.5%) | 93.6% | 0.00 | 5.84% | 14,240,000 | True | False |

## Paired comparisons with completed matching baseline seeds

| Baseline | Success change (pp) | Packet change/episode | Collision step rate change (pp) |
| --- | --- | --- | --- |
| full | +1.40 | -599.79 | +0.20 |
| none | +4.72 | +0.21 | -0.77 |
| fixed_5 | +0.27 | -119.79 | +0.29 |

Across-seed summaries give each training seed equal weight. Sample SDs and 95% percentile bootstrap intervals use training seeds as independent units (20,000 resamples; analysis seed 20261110). Per-policy Wilson intervals describe evaluation episodes only. Partial results are provisional; failed/budget-exhausted target attempts are not hidden or excluded from final evaluation.


## Expanded evaluation provenance

This is the completed 500-episode held-out evaluation. The original 200 scenarios (400000–400199) are retained, and 300 scenarios (400200–400499) are added for each training seed. Frozen checkpoints and checkpoint selection are unchanged; no retraining was performed. Original training artifacts, initialization audits and 200-episode results remain at `results/adaptive-zero-loss-15-seeds-20261110/`.

Current artifacts: `results/scheduler-heldout500-20261112/adaptive/summary.json`, `per-seed.csv`, `evaluation-per-seed.csv`, `episodes.jsonl`, and `within-seed-variance-200-vs-500.csv`. Shared checkpoint/source hashes, raw evaluations and verification: `results/scheduler-heldout500-20261112/`. Initial-none and other baseline references use the completed 500-episode baseline evaluation. The adaptive results in the ablation and warm-none comparison are the same outcomes, not independent reruns.

## Within-seed episode variance and SD (500 episodes)

Coverage SD is in percentage points. These are episode statistics within a training seed; the overall report's seed SDs describe variation of seed-level means.

| Seed | Group | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |
| --- | --- | --- | --- | --- | --- |
| 42 | adaptive | 278.456 | 16.687 | 0.018495 | 13.60 |
| 43 | adaptive | 299.900 | 17.318 | 0.024148 | 15.54 |
| 44 | adaptive | 309.012 | 17.579 | 0.015809 | 12.57 |
| 45 | adaptive | 311.524 | 17.650 | 0.024377 | 15.61 |
| 46 | adaptive | 333.721 | 18.268 | 0.022151 | 14.88 |
| 47 | adaptive | 304.798 | 17.458 | 0.019386 | 13.92 |
| 48 | adaptive | 280.290 | 16.742 | 0.012202 | 11.05 |
| 49 | adaptive | 296.993 | 17.233 | 0.017134 | 13.09 |
| 50 | adaptive | 320.891 | 17.913 | 0.023521 | 15.34 |
| 51 | adaptive | 363.789 | 19.073 | 0.026683 | 16.33 |
| 52 | adaptive | 380.405 | 19.504 | 0.027655 | 16.63 |
| 53 | adaptive | 322.810 | 17.967 | 0.021723 | 14.74 |
| 54 | adaptive | 324.509 | 18.014 | 0.021109 | 14.53 |
| 55 | adaptive | 262.586 | 16.205 | 0.014644 | 12.10 |
| 56 | adaptive | 373.215 | 19.319 | 0.032413 | 18.00 |

## Variance change: 200 versus 500 held-out episodes

Each seed's episode variance is a sample variance (denominators 199 and 499). This table first averages those variances equally across the 15 seeds, then reports 100 × (variance500 − variance200) / variance200. It is not a change in standard error or a mean of seed-specific percentage changes. Coverage variance uses fractions on the 0–1 scale; task-return variance uses squared return units. Individual seed changes can have either sign. More evaluation episodes improve the estimate; they do not inherently reduce outcome variance.

| Group | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| adaptive | task_return | 296.709703 | 317.526628 | +7.02% |
| adaptive | coverage | 0.023385 | 0.021430 | -8.36% |
