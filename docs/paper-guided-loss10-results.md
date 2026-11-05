# Adaptive policy evaluation at 10% packet loss

Completed November 5, 2026. Evaluated the three selected, frozen zero-loss checkpoints on **400 paired physical scenarios per policy**, split into 200 matched and 200 fresh-cohort scenarios: **1,200 episodes total**. No retraining, tuning or checkpoint reselection was performed.

## Combined results

| Policy | Success (95% Wilson interval) | Any collision | Attempts / 600 | Packets / episode | Attempts / active step | Mean steps to termination |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Initial adaptive checkpoint | 54.50% (49.6%-59.3%) | 26.25% | 6.10% | 36.575 | 0.445 | 82.17 |
| Equal-budget joint PPO continuation | 55.00% (50.1%-59.8%) | 26.75% | 5.89% | 35.310 | 0.437 | 80.72 |
| Paper-inspired modular constrained PPO | 52.25% (47.4%-57.1%) | 8.25% | 5.77% | 34.625 | 0.369 | 93.95 |

The modular policy retained the lowest measured collision fraction and attempted traffic, but had lower final full-coverage success than the other two policies. All policies remained below the 20% attempted-packet budget; none reached the >90% success target. These are fixed-checkpoint measurements, not evidence from multiple training seeds.

## Change from zero loss on the same scenarios

| Policy | Success at 0% loss | Success at 10% loss | Change (percentage points) | Collisions at 0% loss | Collisions at 10% loss | Change (percentage points) |
| --- | --- | --- | ---: | --- | --- | ---: |
| Initial adaptive checkpoint | 53.50% | 54.50% | +1.00 | 26.50% | 26.25% | -0.25 |
| Equal-budget joint PPO continuation | 55.75% | 55.00% | -0.75 | 25.75% | 26.75% | +1.00 |
| Paper-inspired modular constrained PPO | 53.00% | 52.25% | -0.75 | 9.50% | 8.25% | -1.25 |

The observed success change was +1.00 percentage point for the initial checkpoint and -0.75 point for each continuation. Small gains under sampled loss should not be interpreted as a general benefit of packet drops: this evaluation uses one seeded channel realization per scenario, with identical per-link/per-step random draws across policies. Episode confidence intervals and paired success/failure changes are saved in the detailed results.

## Results by cohort

| Cohort | Policy | Success | Any collision | Attempts / 600 | Attempts / active step | Mean steps |
| --- | --- | --- | --- | --- | ---: | ---: |
| matched | Initial adaptive checkpoint | 57.0% | 24.0% | 6.02% | 0.426 | 84.75 |
| matched | Equal-budget joint PPO continuation | 54.0% | 28.0% | 5.92% | 0.441 | 80.52 |
| matched | Paper-inspired modular constrained PPO | 50.5% | 9.0% | 5.71% | 0.367 | 93.36 |
| fresh | Initial adaptive checkpoint | 52.0% | 28.5% | 6.17% | 0.466 | 79.58 |
| fresh | Equal-budget joint PPO continuation | 56.0% | 25.5% | 5.86% | 0.434 | 80.93 |
| fresh | Paper-inspired modular constrained PPO | 54.0% | 7.5% | 5.84% | 0.370 | 94.54 |

## Evaluation contract

- Three agents and three static landmarks, 0.4-unit minimum initial pairwise separation for both agents and landmarks, sensing radius 0.6, coverage radius 0.1, and 100 scheduled steps.
- A collision immediately terminates the episode as failure. Success requires no collision throughout the episode and simultaneous full coverage at step 100. No starts or failures were filtered.
- The environment independently drops each attempted directed packet with probability 0.10. Message payloads remain position/velocity (16 bytes), with one-step delivery and five-step cache expiration. Actors receive no new loss-rate feature or acknowledgments.
- Attempted packet ratio is attempts / **600 per scheduled episode**, including attempts on the fatal step and dropped packets. Attempts per active step is total attempts / total executed steps, aggregated across episodes.
- Matched seeds are 400000-400199; fresh-cohort seeds are 600000-600199. These physical scenarios were already evaluated at zero loss; "fresh" refers to their original separation from training/selection.
- Current task-return scoring stops distance costs while full coverage is maintained; collision costs remain. Existing policies were trained under the earlier scoring. Historical zero-loss task returns are not compared across scoring versions.

Across all policies, 4,166 of 42,604 attempted packets were lost (**9.78%**). The configured probability is 10%; the realized fraction varies with sampled links and policy decisions.

## Replays and reproducibility

[Matched comparison](../results/paper-guided-loss10-20261105/matched/comparison.html) | [Fresh comparison](../results/paper-guided-loss10-20261105/fresh/comparison.html) | [Matched all-success example](../results/paper-guided-loss10-20261105/matched/all-success.html) | [Fresh all-success example](../results/paper-guided-loss10-20261105/fresh/all-success.html)

All-success examples select the first scenario successful for all three policies, solely for illustration. Every evaluated scenario contributes to the reported statistics.

[Detailed report](../results/paper-guided-loss10-20261105/report.md), [complete summaries](../results/paper-guided-loss10-20261105/summary.json), [verification and pooled statistics](../results/paper-guided-loss10-20261105/verification.json), [checkpoint provenance](../results/paper-guided-loss10-20261105/checkpoints.json), [manifest](../results/paper-guided-loss10-20261105/manifest.json), and [completed progress page](../results/paper-guided-loss10-20261105/progress.html). Each cohort directory contains per-policy episode JSONL logs.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_paper_loss.py --loss 0.1 --episodes 200 --output results/paper-guided-loss10-new
.\.venv\Scripts\python.exe scripts/verify_paper_loss.py results/paper-guided-loss10-new
```

Verification passed for all 1,200 episode records and all 12 replay policy runs (886 action frames). It checked unchanged checkpoint/source hashes, matched starting layouts, recomputed summary statistics, the fixed communication denominator, task-return accounting and exact seeded directed-link deliveries. A 12-episode zero-loss regression also reproduced every historical non-return episode field; a separate 12-episode 10% smoke pipeline passed the same audit.
