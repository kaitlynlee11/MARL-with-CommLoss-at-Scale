# Adaptive scheduler: 15 seeds, zero communication loss

Status: complete; 15/15 seeds evaluated.

Targets: final-step success strictly above 90% AND mean attempted directed packets strictly below 120 per 100-step episode. Both targets use point estimates. Traffic is an episode average, not a hard cap on individual episodes.

Three agents, three static landmarks, zero communication loss, pairwise radial agent-start and landmark separation >=0.4, sensing radius 0.6 and coverage radius 0.1. Episodes run all 100 steps; collisions are recorded without terminating episodes. Collision step rate is the fraction of steps with any overlapping agent pair. This matches the corrected baseline workflow.

Each seed warm-starts its own full-communication baseline actor (8,960,000 baseline training steps); movement and transmission then learn jointly with a fresh critic and optimizer. Additional adaptive steps are reported separately. Shared base task-return scoring matches the baselines (legacy-distance-v0); adaptive shaping rewards differ. This is not a comparison at equal training compute.

Validation uses 200 batched scenarios (seed 100000). Passing validation triggers separate 200-episode original-environment confirmation (500000-500199). Training stops on confirmed targets or its step ceiling. Final frozen-policy evaluation uses the baseline-matched 200 scenarios (400000-400199); held-out results never select or tune checkpoints. A confirmation pass does not guarantee a held-out pass.

| Completed seeds | Success mean | Success seed SD | Mean packets/episode | Collision step rate | Both targets on held-out |
| --- | --- | --- | --- | --- | --- |
| 15 | 90.27% | 2.00% | 0.23 | 5.00% | 9/15 |

Aggregate point-estimate targets met: **True** (final).

| Metric | Mean | Seed SD | 95% seed bootstrap interval |
| --- | --- | --- | --- |
| success_rate | 0.902667 | 0.019988091692900806 | [0.8926666666666668, 0.9123333333333333] |
| final_coverage | 0.954000 | 0.010231448562277973 | [0.949, 0.959] |
| task_return | -28.615392 | 1.0881599747322874 | [-29.117971927903703, -28.05766585624686] |
| attempted_links | 0.227333 | 0.8804582140378198 | [0.0, 0.682] |
| collision_step_rate | 0.050000 | 0.005309425580983316 | [0.047453333333333333, 0.05257333333333333] |

## Individual training seeds

| Seed | Held-out success (95% Wilson) | Coverage | Packets/episode | Collision step rate | Adaptive steps | Confirmation pass | Held-out pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 89.0% (83.9%-92.6%) | 95.0% | 0.00 | 5.01% | 12,320,000 | True | False |
| 43 | 87.5% (82.2%-91.4%) | 95.0% | 0.00 | 4.25% | 10,720,000 | True | False |
| 44 | 91.0% (86.2%-94.2%) | 95.8% | 0.00 | 4.72% | 14,720,000 | True | True |
| 45 | 88.5% (83.3%-92.2%) | 94.0% | 3.41 | 5.46% | 17,120,000 | True | False |
| 46 | 89.0% (83.9%-92.6%) | 94.7% | 0.00 | 4.84% | 12,000,000 | True | False |
| 47 | 88.0% (82.8%-91.8%) | 94.8% | 0.00 | 4.51% | 12,000,000 | True | False |
| 48 | 92.5% (88.0%-95.4%) | 97.0% | 0.00 | 4.86% | 14,880,000 | True | True |
| 49 | 91.0% (86.2%-94.2%) | 95.8% | 0.00 | 5.04% | 11,680,000 | True | True |
| 50 | 91.5% (86.8%-94.6%) | 96.2% | 0.00 | 4.35% | 11,040,000 | True | True |
| 51 | 92.0% (87.4%-95.0%) | 96.3% | 0.00 | 5.22% | 13,120,000 | True | True |
| 52 | 92.5% (88.0%-95.4%) | 95.7% | 0.00 | 5.86% | 14,080,000 | True | True |
| 53 | 90.5% (85.6%-93.8%) | 95.0% | 0.00 | 5.31% | 12,320,000 | True | True |
| 54 | 90.5% (85.6%-93.8%) | 95.0% | 0.00 | 5.50% | 17,440,000 | True | True |
| 55 | 93.5% (89.2%-96.2%) | 97.2% | 0.00 | 4.24% | 12,320,000 | True | True |
| 56 | 87.0% (81.6%-91.0%) | 93.5% | 0.00 | 5.84% | 14,240,000 | True | False |

## Paired comparisons with completed matching baseline seeds

| Baseline | Success change (pp) | Packet change/episode | Collision step rate change (pp) |
| --- | --- | --- | --- |
| full | +1.93 | -599.77 | +0.23 |
| none | +5.13 | +0.23 | -0.72 |
| fixed_5 | +0.60 | -119.77 | +0.29 |

Across-seed summaries give each training seed equal weight. Sample SDs and 95% percentile bootstrap intervals use training seeds as independent units (20,000 resamples; analysis seed 20261110). Per-policy Wilson intervals describe evaluation episodes only. Partial results are provisional; failed/budget-exhausted target attempts are not hidden or excluded from final evaluation.

Artifacts: per-seed.csv, summary.json, seed-N/training/learning.jsonl, seed-N/training/config.json, checkpoint/initializer hashes, confirmations, frozen evaluation/episodes.jsonl, selection.json and result.json. The live progress.html also opens directly from disk and refreshes every five seconds.
