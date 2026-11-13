# Frozen scheduler comparison under communication loss

Status: complete; 135/135 loss-tested seed-policy conditions, plus 45 loss-invariant no-communication reference conditions.

All four selected zero-loss-trained policies are frozen, with no retraining or checkpoint reselection. Training seeds 42-56; 200 matched scenarios (400000-400199) per seed and policy. Adaptive, full and fixed_5 were evaluated separately at each loss rate. The trained no-communication policy (`none`) reuses its verified zero-loss baseline measurements: with zero attempted packets and no loss-rate observation, changing packet loss cannot change its deterministic trajectory. Its identical rows at 10%, 25% and 50% are loss-invariant references, not additional simulated episodes. Rates are configured and known to the evaluator, but actors have no loss-rate input or acknowledgments. This evaluates robustness of the existing policies, not explicitly loss-aware policies.

Independent Bernoulli loss applies to each attempted directed packet. All policies and loss rates use the same seed-indexed per-link/per-step random draws, including draws for silent senders. Agent starts and landmarks have minimum pairwise radial separation 0.4; horizon is 100, without collision termination. Success is simultaneous full landmark coverage at the final step. Coverage is the final fraction of landmarks covered.

Completed packets means successfully delivered directed packets per episode. Attempted and dropped packets are recorded separately. Collision step rate is the fraction of steps with any overlapping pair; collision-episode fraction is the fraction of episodes with at least one collision. Collision-pair counts per episode are also saved.

## Performance with 95% seed bootstrap intervals

Every metric below is **mean [95% seed bootstrap interval]**, with equal weight for each of 15 independent training seeds. Intervals use the percentile bootstrap of seed-level episode means (20,000 resamples; analysis RNG seed 20261110). They do not treat the 3,000 evaluation episodes as independent training replicates. The `none` rows reuse the same loss-invariant baseline estimates at each loss rate.

Task return is the mean cumulative base task reward over the 100-step episode, using the baseline-matched `legacy-distance-v0` scoring: `-0.5 * sum(nearest-agent distance to each landmark) - overlapping_agent_pairs / 3` per step. It excludes adaptive training bonuses and packet penalties; higher (less negative) is better.

| Loss | Policy | Seeds | Success | Coverage | Task return | Collision step rate | Episodes with collisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | adaptive | 15 | 90.23% [89.23%, 91.20%] | 95.38% [94.87%, 95.89%] | -28.63 [-29.15, -28.08] | 5.01% [4.74%, 5.27%] | 70.07% [68.90%, 71.17%] |
| 10% | full | 15 | 88.57% [86.83%, 90.03%] | 94.57% [93.81%, 95.22%] | -28.02 [-28.79, -27.34] | 4.81% [4.50%, 5.16%] | 66.97% [65.67%, 68.23%] |
| 10% | fixed_5 | 15 | 87.63% [85.73%, 89.47%] | 94.43% [93.41%, 95.39%] | -28.55 [-29.28, -27.81] | 4.86% [4.54%, 5.19%] | 67.47% [66.03%, 68.87%] |
| 10% | none | 15 | 85.13% [82.03%, 88.10%] | 92.64% [91.04%, 94.17%] | -31.14 [-32.60, -29.64] | 5.72% [5.19%, 6.28%] | 73.47% [72.17%, 74.80%] |
| 25% | adaptive | 15 | 90.30% [89.33%, 91.23%] | 95.43% [94.96%, 95.91%] | -28.61 [-29.12, -28.06] | 5.00% [4.74%, 5.26%] | 70.07% [68.90%, 71.17%] |
| 25% | full | 15 | 88.67% [87.27%, 90.07%] | 94.67% [93.93%, 95.38%] | -28.43 [-29.24, -27.65] | 4.89% [4.55%, 5.26%] | 67.13% [65.63%, 68.60%] |
| 25% | fixed_5 | 15 | 85.80% [83.50%, 88.00%] | 93.72% [92.81%, 94.57%] | -29.05 [-29.89, -28.20] | 5.03% [4.69%, 5.38%] | 68.90% [67.20%, 70.27%] |
| 25% | none | 15 | 85.13% [82.03%, 88.10%] | 92.64% [91.04%, 94.17%] | -31.14 [-32.60, -29.64] | 5.72% [5.19%, 6.28%] | 73.47% [72.17%, 74.80%] |
| 50% | adaptive | 15 | 90.33% [89.37%, 91.27%] | 95.47% [95.00%, 95.93%] | -28.59 [-29.11, -28.05] | 5.00% [4.74%, 5.26%] | 70.07% [68.90%, 71.17%] |
| 50% | full | 15 | 88.33% [86.63%, 89.93%] | 94.68% [93.92%, 95.42%] | -28.14 [-28.85, -27.40] | 4.73% [4.46%, 4.99%] | 66.90% [65.43%, 68.33%] |
| 50% | fixed_5 | 15 | 77.03% [72.27%, 81.67%] | 90.10% [88.17%, 91.98%] | -30.47 [-31.26, -29.67] | 5.45% [5.15%, 5.75%] | 69.90% [68.50%, 71.03%] |
| 50% | none | 15 | 85.13% [82.03%, 88.10%] | 92.64% [91.04%, 94.17%] | -31.14 [-32.60, -29.64] | 5.72% [5.19%, 6.28%] | 73.47% [72.17%, 74.80%] |

## Packet counts with 95% seed bootstrap intervals

Counts are directed packets per episode; completed packets are the delivered packets. The fixed schedules have constant attempt counts. Their delivery counts also have degenerate seed intervals because all policies use the same scenario-indexed channel draws across training seeds; these seed intervals do not measure uncertainty over new channel realizations.

| Loss | Policy | Delivered packets/episode | Attempted packets/episode |
| --- | --- | --- | --- |
| 10% | adaptive | 0.21 [0.00, 0.62] | 0.23 [0.00, 0.69] |
| 10% | full | 539.06 [539.06, 539.06] | 600.00 [600.00, 600.00] |
| 10% | fixed_5 | 107.81 [107.81, 107.81] | 120.00 [120.00, 120.00] |
| 10% | none | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 25% | adaptive | 0.18 [0.00, 0.53] | 0.23 [0.00, 0.69] |
| 25% | full | 449.55 [449.55, 449.55] | 600.00 [600.00, 600.00] |
| 25% | fixed_5 | 90.19 [90.19, 90.19] | 120.00 [120.00, 120.00] |
| 25% | none | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 50% | adaptive | 0.12 [0.00, 0.36] | 0.23 [0.00, 0.69] |
| 50% | full | 300.02 [300.02, 300.02] | 600.00 [600.00, 600.00] |
| 50% | fixed_5 | 60.00 [60.00, 60.00] | 120.00 [120.00, 120.00] |
| 50% | none | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Means weight independent training seeds equally. summary.json (and no-communication-reference.json for `none`) includes task-return means, sample SDs, 95% percentile bootstrap intervals across training seeds (20,000 resamples), and paired changes against each policy's original zero-loss results. Wilson intervals in per-seed summaries measure episode uncertainty only. Scenario pairing does not make evaluation episodes independent training replicates.

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
