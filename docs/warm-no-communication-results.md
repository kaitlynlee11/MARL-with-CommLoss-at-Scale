# Warm-start no-communication control versus initial none and adaptive

Status: complete; 15/15 matched policy seeds evaluated.

All comparisons use original training seeds 42-56 and the same 500 held-out scenarios (400000-400499), with three agents/landmarks, zero loss, 0.4-unit radial agent-start and landmark spacing, 100-step episodes and the original legacy communication/task-return protocol. This study is separate from the exploratory landmark-sharing changes.

The new no-communication policy starts from the exact full-communication actor initializer used by adaptive for that seed. Its five-action actor preserves that initializer's movement distribution. To match the fresh adaptive critic exactly, the same seeded ten-action model is constructed first, then only its actor is replaced with the initializer's five-action movement actor. The critic, optimizer and PopArt state are fresh. Saved initial.pt verifies actor equality, fresh-critic equality and an empty optimizer.

Each control trains with zero transmit gates throughout, the adaptive navigation reward (coverage/full-coverage bonuses, collision/time/distance terms; packet term is zero), and the identical PPO hyperparameters, batch/rollout/sequence sizes and number of epochs. Each seed's actual adaptive steps and optimizer update count are matched, rather than the unused 100-million-step ceiling. Early stopping is disabled. The primary control policy is final last.pt at the matched budget, matching adaptive's final target.pt update; validation-best artifacts are retained but not used to select the reported control. Held-out results never select checkpoints.

Initial none is the separately trained from-scratch 8,960,000-step baseline. Warm none and adaptive additionally share the same initializer-training cost and seed-specific optimization work. Initial none is an original reference, not an equal-compute condition. The movement-only control has five actions, while adaptive has ten joint movement/transmit actions; no dummy transmit actions or extra exploration choices are added to the control.

Every entry is mean [95% training-seed bootstrap interval], using 20,000 percentile resamples of equally weighted seeds. Partial summaries compare only matching completed seeds across all three groups. Task return uses baseline legacy-distance-v0 scoring, excludes communication penalties and shaping bonuses, and is better when less negative.

| Policy | Seeds | Success | Coverage | Collision step rate | Task return | Packets/episode |
| --- | --- | --- | --- | --- | --- | --- |
| initial_none | 15 | 85.97% [83.19%, 88.75%] | 93.16% [91.83%, 94.49%] | 5.83% [5.38%, 6.27%] | -30.9112 [-32.1008, -29.6667] | 0.0000 |
| warm_none | 15 | 89.20% [88.24%, 90.09%] | 95.14% [94.72%, 95.56%] | 4.81% [4.54%, 5.07%] | -28.2994 [-28.7772, -27.8052] | 0.0000 |
| adaptive | 15 | 90.69% [89.83%, 91.51%] | 95.70% [95.23%, 96.14%] | 5.06% [4.83%, 5.31%] | -28.6712 [-29.0939, -28.2792] | 0.2099 |

## Paired differences (warm none minus reference)

| Reference | Success change (pp) | Collision step rate change (pp) | Return change |
| --- | --- | --- | --- |
| warm_none_minus_initial_none | +3.2267 [+0.2667, +6.2933] | -1.0253 [-1.4768, -0.5563] | +2.6118 [+1.2935, +3.8523] |
| warm_none_minus_adaptive | -1.4933 [-2.8933, -0.1333] | -0.2529 [-0.4865, -0.0360] | +0.3717 [+0.0193, +0.7359] |

## Seed-specific optimization budgets

Training budgets and initialization audits are unchanged from the archived training study. These totals exclude held-out evaluation steps.

| Seed | Additional world steps | PPO updates | Optimizer steps | Total work including initializer |
| --- | --- | --- | --- | --- |
| 42 | 12,320,000 | 1,925 | 23,100 | 21,280,000 |
| 43 | 10,720,000 | 1,675 | 20,100 | 19,680,000 |
| 44 | 14,720,000 | 2,300 | 27,600 | 23,680,000 |
| 45 | 17,120,000 | 2,675 | 32,100 | 26,080,000 |
| 46 | 12,000,000 | 1,875 | 22,500 | 20,960,000 |
| 47 | 12,000,000 | 1,875 | 22,500 | 20,960,000 |
| 48 | 14,880,000 | 2,325 | 27,900 | 23,840,000 |
| 49 | 11,680,000 | 1,825 | 21,900 | 20,640,000 |
| 50 | 11,040,000 | 1,725 | 20,700 | 20,000,000 |
| 51 | 13,120,000 | 2,050 | 24,600 | 22,080,000 |
| 52 | 14,080,000 | 2,200 | 26,400 | 23,040,000 |
| 53 | 12,320,000 | 1,925 | 23,100 | 21,280,000 |
| 54 | 17,440,000 | 2,725 | 32,700 | 26,400,000 |
| 55 | 12,320,000 | 1,925 | 23,100 | 21,280,000 |
| 56 | 14,240,000 | 2,225 | 26,700 | 23,200,000 |


## Expanded evaluation provenance

This is the completed 500-episode held-out evaluation. The original 200 scenarios (400000–400199) are retained, and 300 scenarios (400200–400499) are added for each training seed. Frozen checkpoints and checkpoint selection are unchanged; no retraining was performed. Original training artifacts, initialization audits and 200-episode results remain at `results/warm-no-communication-15-seeds-20261112/`.

Current artifacts: `results/scheduler-heldout500-20261112/warm-none/summary.json`, `per-seed.csv`, `evaluation-per-seed.csv`, `episodes.jsonl`, and `within-seed-variance-200-vs-500.csv`. Shared checkpoint/source hashes, raw evaluations and verification: `results/scheduler-heldout500-20261112/`. Initial-none and other baseline references use the completed 500-episode baseline evaluation. The adaptive results in the ablation and warm-none comparison are the same outcomes, not independent reruns.

## Individual held-out evaluations

| Seed | Group | Success | Coverage | Task return | Collision step rate | Packets/episode |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | initial_none | 88.00% | 93.53% | -31.0260 | 5.432% | 0.0000 |
| 42 | warm_none | 89.20% | 95.07% | -27.7955 | 4.608% | 0.0000 |
| 42 | adaptive | 90.20% | 95.80% | -28.2500 | 4.732% | 0.0000 |
| 43 | initial_none | 93.60% | 96.80% | -28.2827 | 4.738% | 0.0000 |
| 43 | warm_none | 90.80% | 96.20% | -26.3086 | 3.776% | 0.0000 |
| 43 | adaptive | 88.80% | 94.93% | -28.0511 | 4.450% | 0.0000 |
| 44 | initial_none | 84.20% | 92.33% | -31.4256 | 5.902% | 0.0000 |
| 44 | warm_none | 88.60% | 94.73% | -27.1375 | 4.186% | 0.0000 |
| 44 | adaptive | 92.40% | 96.67% | -28.0019 | 4.800% | 0.0000 |
| 45 | initial_none | 75.60% | 87.87% | -35.2222 | 7.356% | 0.0000 |
| 45 | warm_none | 91.00% | 96.33% | -28.2597 | 5.020% | 0.0000 |
| 45 | adaptive | 90.60% | 95.40% | -28.2169 | 5.244% | 3.1480 |
| 46 | initial_none | 85.80% | 93.00% | -31.6220 | 6.322% | 0.0000 |
| 46 | warm_none | 89.80% | 95.67% | -29.3776 | 4.690% | 0.0000 |
| 46 | adaptive | 89.80% | 95.40% | -28.7684 | 4.852% | 0.0000 |
| 47 | initial_none | 87.20% | 93.80% | -31.4903 | 6.082% | 0.0000 |
| 47 | warm_none | 90.60% | 94.87% | -28.1208 | 4.710% | 0.0000 |
| 47 | adaptive | 90.60% | 95.80% | -28.0264 | 4.434% | 0.0000 |
| 48 | initial_none | 89.80% | 95.20% | -29.4699 | 4.970% | 0.0000 |
| 48 | warm_none | 89.60% | 95.53% | -28.7880 | 5.200% | 0.0000 |
| 48 | adaptive | 93.60% | 97.33% | -28.3564 | 4.712% | 0.0000 |
| 49 | initial_none | 94.80% | 97.27% | -26.2698 | 4.448% | 0.0000 |
| 49 | warm_none | 85.80% | 93.87% | -29.2607 | 5.230% | 0.0000 |
| 49 | adaptive | 91.60% | 96.33% | -28.3963 | 4.932% | 0.0000 |
| 50 | initial_none | 93.20% | 96.80% | -26.6536 | 4.462% | 0.0000 |
| 50 | warm_none | 92.40% | 96.20% | -27.6179 | 4.526% | 0.0000 |
| 50 | adaptive | 89.40% | 95.20% | -28.8365 | 4.880% | 0.0000 |
| 51 | initial_none | 84.80% | 93.00% | -31.1535 | 5.794% | 0.0000 |
| 51 | warm_none | 89.80% | 95.67% | -28.6643 | 4.444% | 0.0000 |
| 51 | adaptive | 89.40% | 94.80% | -30.0237 | 5.772% | 0.0000 |
| 52 | initial_none | 78.40% | 90.20% | -33.7331 | 7.112% | 0.0000 |
| 52 | warm_none | 90.20% | 94.87% | -29.4128 | 5.880% | 0.0000 |
| 52 | adaptive | 91.20% | 95.33% | -29.6254 | 5.934% | 0.0000 |
| 53 | initial_none | 86.00% | 92.33% | -30.9579 | 6.094% | 0.0000 |
| 53 | warm_none | 85.40% | 94.07% | -28.4753 | 4.744% | 0.0000 |
| 53 | adaptive | 91.60% | 95.93% | -29.1544 | 5.526% | 0.0000 |
| 54 | initial_none | 82.40% | 92.07% | -30.8486 | 5.978% | 0.0000 |
| 54 | warm_none | 87.80% | 94.87% | -28.4463 | 5.086% | 0.0000 |
| 54 | adaptive | 91.60% | 96.00% | -28.9690 | 5.210% | 0.0000 |
| 55 | initial_none | 78.80% | 89.40% | -34.0929 | 7.030% | 0.0000 |
| 55 | warm_none | 89.20% | 95.60% | -26.9895 | 4.358% | 0.0000 |
| 55 | adaptive | 92.80% | 96.93% | -27.1213 | 4.578% | 0.0000 |
| 56 | initial_none | 87.00% | 93.80% | -31.4205 | 5.760% | 0.0000 |
| 56 | warm_none | 87.80% | 93.60% | -29.8370 | 5.642% | 0.0000 |
| 56 | adaptive | 86.80% | 93.60% | -30.2698 | 5.838% | 0.0000 |

## Within-seed episode variance and SD (500 episodes)

Coverage SD is in percentage points. These are episode statistics within a training seed; the overall report's seed SDs describe variation of seed-level means.

| Seed | Group | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |
| --- | --- | --- | --- | --- | --- |
| 42 | adaptive | 278.456 | 16.687 | 0.018495 | 13.60 |
| 42 | warm_none | 299.385 | 17.303 | 0.023391 | 15.29 |
| 42 | initial_none | 393.689 | 19.842 | 0.037449 | 19.35 |
| 43 | adaptive | 299.900 | 17.318 | 0.024148 | 15.54 |
| 43 | warm_none | 222.011 | 14.900 | 0.016589 | 12.88 |
| 43 | initial_none | 309.568 | 17.595 | 0.017678 | 13.30 |
| 44 | adaptive | 309.012 | 17.579 | 0.015809 | 12.57 |
| 44 | warm_none | 242.961 | 15.587 | 0.026390 | 16.25 |
| 44 | initial_none | 355.212 | 18.847 | 0.037531 | 19.37 |
| 45 | adaptive | 311.524 | 17.650 | 0.024377 | 15.61 |
| 45 | warm_none | 316.835 | 17.800 | 0.015353 | 12.39 |
| 45 | initial_none | 497.105 | 22.296 | 0.057393 | 23.96 |
| 46 | adaptive | 333.721 | 18.268 | 0.022151 | 14.88 |
| 46 | warm_none | 314.690 | 17.740 | 0.019272 | 13.88 |
| 46 | initial_none | 498.607 | 22.330 | 0.035838 | 18.93 |
| 47 | adaptive | 304.798 | 17.458 | 0.019386 | 13.92 |
| 47 | warm_none | 305.927 | 17.491 | 0.029646 | 17.22 |
| 47 | initial_none | 380.340 | 19.502 | 0.031998 | 17.89 |
| 48 | adaptive | 280.290 | 16.742 | 0.012202 | 11.05 |
| 48 | warm_none | 325.463 | 18.041 | 0.020045 | 14.16 |
| 48 | initial_none | 338.677 | 18.403 | 0.024857 | 15.77 |
| 49 | adaptive | 296.993 | 17.233 | 0.017134 | 13.09 |
| 49 | warm_none | 339.549 | 18.427 | 0.026959 | 16.42 |
| 49 | initial_none | 253.585 | 15.924 | 0.017287 | 13.15 |
| 50 | adaptive | 320.891 | 17.913 | 0.023521 | 15.34 |
| 50 | warm_none | 287.270 | 16.949 | 0.021043 | 14.51 |
| 50 | initial_none | 289.004 | 17.000 | 0.016342 | 12.78 |
| 51 | adaptive | 363.789 | 19.073 | 0.026683 | 16.33 |
| 51 | warm_none | 285.292 | 16.891 | 0.019717 | 14.04 |
| 51 | initial_none | 352.627 | 18.778 | 0.032276 | 17.97 |
| 52 | adaptive | 380.405 | 19.504 | 0.027655 | 16.63 |
| 52 | warm_none | 356.353 | 18.877 | 0.028310 | 16.83 |
| 52 | initial_none | 426.932 | 20.662 | 0.041368 | 20.34 |
| 53 | adaptive | 322.810 | 17.967 | 0.021723 | 14.74 |
| 53 | warm_none | 286.425 | 16.924 | 0.023861 | 15.45 |
| 53 | initial_none | 390.641 | 19.765 | 0.042875 | 20.71 |
| 54 | adaptive | 324.509 | 18.014 | 0.021109 | 14.53 |
| 54 | warm_none | 311.356 | 17.645 | 0.022521 | 15.01 |
| 54 | initial_none | 343.546 | 18.535 | 0.035778 | 18.92 |
| 55 | adaptive | 262.586 | 16.205 | 0.014644 | 12.10 |
| 55 | warm_none | 245.100 | 15.656 | 0.018100 | 13.45 |
| 55 | initial_none | 443.214 | 21.053 | 0.049975 | 22.36 |
| 56 | adaptive | 373.215 | 19.319 | 0.032413 | 18.00 |
| 56 | warm_none | 373.570 | 19.328 | 0.035085 | 18.73 |
| 56 | initial_none | 398.567 | 19.964 | 0.030662 | 17.51 |

## Variance change: 200 versus 500 held-out episodes

Each seed's episode variance is a sample variance (denominators 199 and 499). This table first averages those variances equally across the 15 seeds, then reports 100 × (variance500 − variance200) / variance200. It is not a change in standard error or a mean of seed-specific percentage changes. Coverage variance uses fractions on the 0–1 scale; task-return variance uses squared return units. Individual seed changes can have either sign. More evaluation episodes improve the estimate; they do not inherently reduce outcome variance.

| Group | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| adaptive | task_return | 296.709703 | 317.526628 | +7.02% |
| adaptive | coverage | 0.023385 | 0.021430 | -8.36% |
| warm_none | task_return | 273.712187 | 300.812476 | +9.90% |
| warm_none | coverage | 0.022816 | 0.023085 | +1.18% |
| initial_none | task_return | 364.434267 | 378.087675 | +3.75% |
| initial_none | coverage | 0.036627 | 0.033954 | -7.30% |
