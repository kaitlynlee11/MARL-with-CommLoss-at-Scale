# Frozen communication policies: larger-team tests at zero loss

Status: complete; 180/180 seed-policy-team conditions available.

Policies: adaptive, full, fixed_5 and separately trained none. The selected N=3 checkpoints for training seeds 42-56 are frozen and evaluated at N=5, 10 and 20, with the same number of landmarks. There is no larger-team retraining or checkpoint reselection. The shared decentralized actors still receive 28 local values: self position/velocity, up to three visible peers and landmarks, and pooled inbox. The centralized N=3 critic is not used for inference.

Zero packet loss; sensing radius 0.6, coverage radius 0.1, 100-step horizon, collision termination disabled. Both agent starts and landmark centers obey pairwise radial separation >=0.4 in the original [-1,1] square. Landmarks are static. Success requires simultaneous coverage of all N landmarks at the final step. Collision step rate counts steps with any overlapping agent pair; collision-pair counts per step are also saved because step rates may saturate for large teams.

Reset sampling: sequential rejection with bounded retries and whole-layout restarts, followed by label permutation. This is used consistently for all three requested team sizes. It preserves the bounds and minimum-distance rules but changes the reset distribution from the historical N=3 whole-configuration rejection sampler, which is impractical at N=20. No held-out scenario seeds are dropped, replaced or filtered. layouts.json records and hashes all initial layouts; every model and training seed is checked against them.

Each condition uses 200 matched scenarios, seeds 400000-400199; scenario labels pair policies within a team size, not physical layouts across different team sizes. Base task-return scoring is the baseline-matched legacy-distance-v0: -0.5 * sum of nearest-agent landmark distances - overlapping pairs / N per step. Raw returns scale with team size; return per agent is saved as an additional diagnostic, not a claim of identical task difficulty.

Performance metrics and collision-episode fractions below are mean [95% seed bootstrap interval], using equally weighted training seeds and 20,000 percentile-bootstrap resamples. Attempted packets, delivered packets and full-traffic ratios show means only. Partial results are provisional. These tests measure transfer of the trained N=3 policies, not performance of policies trained for each larger environment.

| Agents / landmarks | Policy | Seeds | Success | Coverage | Task return | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- |
| 5/5 | adaptive | 15 | 73.93% [72.07%, 75.87%] | 90.19% [89.15%, 91.10%] | -45.32 [-46.28, -44.40] | 15.83% [15.07%, 16.63%] |
| 5/5 | full | 15 | 70.53% [67.43%, 73.57%] | 88.82% [87.42%, 90.10%] | -46.13 [-47.20, -45.00] | 16.79% [15.97%, 17.62%] |
| 5/5 | fixed_5 | 15 | 73.03% [70.73%, 75.43%] | 89.65% [88.59%, 90.75%] | -45.90 [-46.97, -44.86] | 16.38% [15.60%, 17.23%] |
| 5/5 | none | 15 | 67.80% [64.23%, 71.43%] | 87.09% [85.78%, 88.52%] | -47.59 [-48.94, -46.04] | 16.90% [15.96%, 17.77%] |
| 10/10 | adaptive | 15 | 28.13% [25.17%, 31.27%] | 76.52% [75.32%, 77.72%] | -86.70 [-88.64, -84.75] | 51.26% [49.76%, 52.72%] |
| 10/10 | full | 15 | 20.00% [16.10%, 23.30%] | 70.11% [67.46%, 72.43%] | -92.64 [-95.86, -89.90] | 54.77% [52.84%, 56.82%] |
| 10/10 | fixed_5 | 15 | 23.03% [19.90%, 25.93%] | 71.86% [69.48%, 73.78%] | -92.20 [-95.49, -89.72] | 54.33% [52.78%, 56.09%] |
| 10/10 | none | 15 | 17.93% [14.63%, 21.20%] | 67.68% [65.33%, 70.14%] | -95.63 [-98.38, -92.68] | 55.83% [54.02%, 57.44%] |
| 20/20 | adaptive | 15 | 0.80% [0.33%, 1.37%] | 51.08% [48.64%, 53.45%] | -179.12 [-184.93, -173.56] | 92.78% [91.71%, 93.88%] |
| 20/20 | full | 15 | 0.30% [0.10%, 0.57%] | 42.88% [39.30%, 46.12%] | -198.68 [-209.16, -190.29] | 95.10% [94.31%, 95.86%] |
| 20/20 | fixed_5 | 15 | 0.33% [0.13%, 0.53%] | 44.18% [40.44%, 47.39%] | -197.10 [-208.75, -187.64] | 94.76% [93.83%, 95.64%] |
| 20/20 | none | 15 | 0.13% [0.00%, 0.30%] | 39.99% [36.66%, 43.57%] | -206.58 [-215.36, -197.86] | 95.96% [95.10%, 96.66%] |

| Agents / landmarks | Policy | Seeds | Attempted packets/episode | Delivered packets/episode | Full-traffic ratio | Episodes with collisions |
| --- | --- | --- | --- | --- | --- | --- |
| 5/5 | adaptive | 15 | 0.43 | 0.43 | 0.02% | 92.50% [91.77%, 93.23%] |
| 5/5 | full | 15 | 2000.00 | 2000.00 | 100.00% | 93.43% [92.80%, 94.03%] |
| 5/5 | fixed_5 | 15 | 400.00 | 400.00 | 20.00% | 92.77% [92.17%, 93.33%] |
| 5/5 | none | 15 | 0.00 | 0.00 | 0.00% | 93.80% [92.93%, 94.50%] |
| 10/10 | adaptive | 15 | 0.57 | 0.57 | 0.01% | 100.00% [100.00%, 100.00%] |
| 10/10 | full | 15 | 9000.00 | 9000.00 | 100.00% | 100.00% [100.00%, 100.00%] |
| 10/10 | fixed_5 | 15 | 1800.00 | 1800.00 | 20.00% | 99.97% [99.90%, 100.00%] |
| 10/10 | none | 15 | 0.00 | 0.00 | 0.00% | 100.00% [100.00%, 100.00%] |
| 20/20 | adaptive | 15 | 0.04 | 0.04 | 0.00% | 100.00% [100.00%, 100.00%] |
| 20/20 | full | 15 | 38000.00 | 38000.00 | 100.00% | 100.00% [100.00%, 100.00%] |
| 20/20 | fixed_5 | 15 | 7600.00 | 7600.00 | 20.00% | 100.00% [100.00%, 100.00%] |
| 20/20 | none | 15 | 0.00 | 0.00 | 0.00% | 100.00% [100.00%, 100.00%] |

Full traffic is N*(N-1)*100 directed packets per episode: 2,000 at N=5, 9,000 at N=10, and 38,000 at N=20. Fixed_5 attempts one-fifth of these (400 / 1,800 / 7,600); none attempts zero. Attempted and delivered counts must match at zero loss.

Artifacts: per-seed.csv, summary.json, n-N/seed-S/episodes.jsonl, n-N/seed-S/summary.json, layouts.json, manifest.json, status.json and progress.html. On completion, episodes.jsonl concatenates the complete original-environment evaluation records. Existing N=3 baseline and packet-loss artifacts remain separate.
