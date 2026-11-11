# Frozen scheduler comparison under communication loss

Status: complete; 135/135 loss-tested seed-policy conditions, plus 45 loss-invariant no-communication reference conditions.

All four selected zero-loss-trained policies are frozen, with no retraining or checkpoint reselection. Training seeds 42-56; 200 matched scenarios (400000-400199) per seed and policy. Adaptive, full and fixed_5 were evaluated separately at each loss rate. The trained no-communication policy (`none`) reuses its verified zero-loss baseline measurements: with zero attempted packets and no loss-rate observation, changing packet loss cannot change its deterministic trajectory. Its identical rows at 10%, 25% and 50% are loss-invariant references, not additional simulated episodes. Rates are configured and known to the evaluator, but actors have no loss-rate input or acknowledgments. This evaluates robustness of the existing policies, not explicitly loss-aware policies.

Independent Bernoulli loss applies to each attempted directed packet. All policies and loss rates use the same seed-indexed per-link/per-step random draws, including draws for silent senders. Agent starts and landmarks have minimum pairwise radial separation 0.4; horizon is 100, without collision termination. Success is simultaneous full landmark coverage at the final step. Coverage is the final fraction of landmarks covered.

Completed packets means successfully delivered directed packets per episode. Attempted and dropped packets are recorded separately. Collision step rate is the fraction of steps with any overlapping pair; collision-episode fraction is the fraction of episodes with at least one collision. Collision-pair counts per episode are also saved.

| Loss | Policy | Seeds | Success | Coverage | Collision step rate | Episodes with collisions | Delivered packets/episode | Attempted packets/episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | adaptive | 15 | 90.23% | 95.38% | 5.01% | 70.07% | 0.21 | 0.23 |
| 10% | full | 15 | 88.57% | 94.57% | 4.81% | 66.97% | 539.06 | 600.00 |
| 10% | fixed_5 | 15 | 87.63% | 94.43% | 4.86% | 67.47% | 107.81 | 120.00 |
| 10% | none | 15 | 85.13% | 92.64% | 5.72% | 73.47% | 0.00 | 0.00 |
| 25% | adaptive | 15 | 90.30% | 95.43% | 5.00% | 70.07% | 0.18 | 0.23 |
| 25% | full | 15 | 88.67% | 94.67% | 4.89% | 67.13% | 449.55 | 600.00 |
| 25% | fixed_5 | 15 | 85.80% | 93.72% | 5.03% | 68.90% | 90.19 | 120.00 |
| 25% | none | 15 | 85.13% | 92.64% | 5.72% | 73.47% | 0.00 | 0.00 |
| 50% | adaptive | 15 | 90.33% | 95.47% | 5.00% | 70.07% | 0.12 | 0.23 |
| 50% | full | 15 | 88.33% | 94.68% | 4.73% | 66.90% | 300.02 | 600.00 |
| 50% | fixed_5 | 15 | 77.03% | 90.10% | 5.45% | 69.90% | 60.00 | 120.00 |
| 50% | none | 15 | 85.13% | 92.64% | 5.72% | 73.47% | 0.00 | 0.00 |

Means weight independent training seeds equally. summary.json includes sample SDs, 95% percentile bootstrap intervals across training seeds (20,000 resamples), and paired changes against each policy's original zero-loss results. Wilson intervals in per-seed summaries measure episode uncertainty only. Scenario pairing does not make evaluation episodes independent training replicates.

One seeded channel realization is used per physical scenario. Nearly silent adaptive policies may be insensitive to loss because they do not rely on delivered packets; zero-loss and loss-tested trajectories are verified identical whenever the zero-loss trajectory attempts no packets. No general benefit of packet loss is implied by small observed improvements.

Individual-seed results: per-seed.csv and seed-N/summary.json. Episode records: episodes.jsonl and seed-N/episodes.jsonl. Checkpoint/source hashes and settings: manifest.json. Live completion: progress.html and status.json.

No-communication reference provenance: `results/baseline-agent-spacing04-15-seeds-20261109/seed-N/none/evaluation/episodes.jsonl` (3,000 original baseline episodes). All 15 checkpoints, configurations, matched scenario seeds, 0.4-unit agent and landmark spacing, and zero attempted/delivered packets were checked. `no-communication-reference.json` in the loss-comparison results directory records per-seed summaries, source hashes, and across-seed statistics for these additional rows. The original 27,000-episode loss-test data and its CSV are unchanged.

## Observed comparison

- Adaptive success remained 90.23%-90.33%, with about 0.23 attempted packets per episode. Most selected adaptive policies are silent on this cohort; the results do not show explicit adaptation to a known loss rate.
- Full communication retained 88.33%-88.67% success and the lowest mean collision step rate in each tested loss condition (4.73%-4.89%).
- Fixed-five-step success fell from 87.63% at 10% loss to 77.03% at 50% loss; coverage fell from 94.43% to 90.10%.
- The separately trained no-communication policy retained 85.13% success, 92.64% coverage and a 5.72% collision step rate at every loss rate, with zero packets. At 50% loss it exceeded fixed_5 success by 8.10 percentage points, while fixed_5 had a lower collision step rate. The nearly silent adaptive policies are distinct trained policies with different initialization and additional training; they are not the trained `none` baseline.
- These are fixed-cohort point estimates across the selected seeds, not a statistical-significance or equal-compute algorithm-superiority claim.

[Interactive comparison of the original three loss-tested policies](http://127.0.0.1:8765/loss/comparison.html) includes their zero-loss references and seed interval tooltips. The additional trained `none` reference is included in the table above. Independent verification passed for all 27,000 loss-test episodes, including 18,000 exact fixed-schedule delivery checks; see `verification.json`.
