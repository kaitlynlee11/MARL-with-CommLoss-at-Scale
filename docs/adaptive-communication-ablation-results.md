# Adaptive-policy communication ablation

Status: complete; 15/15 policy seeds evaluated.

The same selected adaptive checkpoint is evaluated normally and with every agent's transmit gate forcibly set to zero. Movement is taken from the original deterministic 10-action joint argmax; only its transmit bit is discarded. The actor and recurrent memory continue operating on the resulting local observations. No retraining, checkpoint reselection, action-probability renormalization or substitution of a separately trained none policy occurs.

Training seeds 42-56; 500 matched held-out scenarios (400000-400499) per policy seed; N=3 agents/landmarks; zero loss; radial starting-agent and landmark separation >=0.4; 100 steps without collision termination. The original frozen simulator, whole-layout sampler and legacy-distance-v0 task reward are preserved with source hashes. This is the original position/velocity-only all-to-all protocol, separate from the exploratory landmark-sharing extension. Original saved episode outcomes are preserved; each normal and disabled pair uses exactly the same scenario.

Collision step rate is the fraction of steps with any overlapping agent pair; collision-episode fraction counts episodes with any collision. Task return is the undiscounted base task reward, excluding packet penalties and training bonuses. Higher (less negative) return is better.

Entries are mean [95% training-seed bootstrap interval]. Differences are **disabled minus original**, bootstrapped directly from the 15 paired seed differences (20,000 percentile resamples; analysis RNG seed 20261110). Rate differences are percentage points. Partial results are provisional.

| Metric | Original adaptive | Communication disabled | Paired difference |
| --- | --- | --- | --- |
| Final success | 90.693% [89.827%, 91.507%] | 90.600% [89.720%, 91.453%] | -0.093 pp [-0.280 pp, +0.000 pp] |
| Collision step rate | 5.060% [4.825%, 5.313%] | 5.087% [4.842%, 5.350%] | +0.028 pp [+0.000 pp, +0.083 pp] |
| Episodes with collisions | 69.920% [68.520%, 71.307%] | 69.960% [68.533%, 71.360%] | +0.040 pp [+0.000 pp, +0.120 pp] |
| Task return | -28.6712 [-29.0939, -28.2792] | -28.7557 [-29.1784, -28.3475] | -0.0845 [-0.2536, +0.0000] |
| Final coverage | 95.698% [95.231%, 96.142%] | 95.631% [95.142%, 96.107%] | -0.067 pp [-0.200 pp, +0.000 pp] |
| Attempted packets/episode | 0.2099 [0.0000, 0.6296] | 0.0000 [0.0000, 0.0000] | -0.2099 [-0.6296, +0.0000] |
| Delivered packets/episode | 0.2099 [0.0000, 0.6296] | 0.0000 [0.0000, 0.0000] | -0.2099 [-0.6296, +0.0000] |

## Per-policy seed comparison

| Seed | Original success | Disabled success | Original collision step rate | Disabled collision step rate | Original return | Disabled return | Original packets/episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 90.2% | 90.2% | 4.732% | 4.732% | -28.2500 | -28.2500 | 0.00 |
| 43 | 88.8% | 88.8% | 4.450% | 4.450% | -28.0511 | -28.0511 | 0.00 |
| 44 | 92.4% | 92.4% | 4.800% | 4.800% | -28.0019 | -28.0019 | 0.00 |
| 45 | 90.6% | 89.2% | 5.244% | 5.660% | -28.2169 | -29.4850 | 3.15 |
| 46 | 89.8% | 89.8% | 4.852% | 4.852% | -28.7684 | -28.7684 | 0.00 |
| 47 | 90.6% | 90.6% | 4.434% | 4.434% | -28.0264 | -28.0264 | 0.00 |
| 48 | 93.6% | 93.6% | 4.712% | 4.712% | -28.3564 | -28.3564 | 0.00 |
| 49 | 91.6% | 91.6% | 4.932% | 4.932% | -28.3963 | -28.3963 | 0.00 |
| 50 | 89.4% | 89.4% | 4.880% | 4.880% | -28.8365 | -28.8365 | 0.00 |
| 51 | 89.4% | 89.4% | 5.772% | 5.772% | -30.0237 | -30.0237 | 0.00 |
| 52 | 91.2% | 91.2% | 5.934% | 5.934% | -29.6254 | -29.6254 | 0.00 |
| 53 | 91.6% | 91.6% | 5.526% | 5.526% | -29.1544 | -29.1544 | 0.00 |
| 54 | 91.6% | 91.6% | 5.210% | 5.210% | -28.9690 | -28.9690 | 0.00 |
| 55 | 92.8% | 92.8% | 4.578% | 4.578% | -27.1213 | -27.1213 | 0.00 |
| 56 | 86.8% | 86.8% | 5.838% | 5.838% | -30.2698 | -30.2698 | 0.00 |

Original episodes with any transmission: 204. Paired success-to-failure changes: 22; failure-to-success changes: 15.

An original episode with zero transmissions must reproduce exactly under forced silence. More generally, movement may change after messages are removed because the recurrent actor receives different inputs; preserving the frozen movement policy does not mean forcing its original action trajectory.


## Expanded evaluation provenance

This is the completed 500-episode held-out evaluation. The original 200 scenarios (400000–400199) are retained, and 300 scenarios (400200–400499) are added for each training seed. Frozen checkpoints and checkpoint selection are unchanged; no retraining was performed. Original training artifacts, initialization audits and 200-episode results remain at `results/adaptive-communication-ablation-20261112/`.

Current artifacts: `results/scheduler-heldout500-20261112/ablation/summary.json`, `per-seed.csv`, `evaluation-per-seed.csv`, `episodes.jsonl`, and `within-seed-variance-200-vs-500.csv`. Shared checkpoint/source hashes, raw evaluations and verification: `results/scheduler-heldout500-20261112/`. Initial-none and other baseline references use the completed 500-episode baseline evaluation. The adaptive results in the ablation and warm-none comparison are the same outcomes, not independent reruns.

## Within-seed episode variance and SD (500 episodes)

Coverage SD is in percentage points. These are episode statistics within a training seed; the overall report's seed SDs describe variation of seed-level means.

| Seed | Group | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |
| --- | --- | --- | --- | --- | --- |
| 42 | adaptive | 278.456 | 16.687 | 0.018495 | 13.60 |
| 42 | communication_disabled | 278.456 | 16.687 | 0.018495 | 13.60 |
| 43 | adaptive | 299.900 | 17.318 | 0.024148 | 15.54 |
| 43 | communication_disabled | 299.900 | 17.318 | 0.024148 | 15.54 |
| 44 | adaptive | 309.012 | 17.579 | 0.015809 | 12.57 |
| 44 | communication_disabled | 309.012 | 17.579 | 0.015809 | 12.57 |
| 45 | adaptive | 311.524 | 17.650 | 0.024377 | 15.61 |
| 45 | communication_disabled | 379.237 | 19.474 | 0.029813 | 17.27 |
| 46 | adaptive | 333.721 | 18.268 | 0.022151 | 14.88 |
| 46 | communication_disabled | 333.721 | 18.268 | 0.022151 | 14.88 |
| 47 | adaptive | 304.798 | 17.458 | 0.019386 | 13.92 |
| 47 | communication_disabled | 304.798 | 17.458 | 0.019386 | 13.92 |
| 48 | adaptive | 280.290 | 16.742 | 0.012202 | 11.05 |
| 48 | communication_disabled | 280.290 | 16.742 | 0.012202 | 11.05 |
| 49 | adaptive | 296.993 | 17.233 | 0.017134 | 13.09 |
| 49 | communication_disabled | 296.993 | 17.233 | 0.017134 | 13.09 |
| 50 | adaptive | 320.891 | 17.913 | 0.023521 | 15.34 |
| 50 | communication_disabled | 320.891 | 17.913 | 0.023521 | 15.34 |
| 51 | adaptive | 363.789 | 19.073 | 0.026683 | 16.33 |
| 51 | communication_disabled | 363.789 | 19.073 | 0.026683 | 16.33 |
| 52 | adaptive | 380.405 | 19.504 | 0.027655 | 16.63 |
| 52 | communication_disabled | 380.405 | 19.504 | 0.027655 | 16.63 |
| 53 | adaptive | 322.810 | 17.967 | 0.021723 | 14.74 |
| 53 | communication_disabled | 322.810 | 17.967 | 0.021723 | 14.74 |
| 54 | adaptive | 324.509 | 18.014 | 0.021109 | 14.53 |
| 54 | communication_disabled | 324.509 | 18.014 | 0.021109 | 14.53 |
| 55 | adaptive | 262.586 | 16.205 | 0.014644 | 12.10 |
| 55 | communication_disabled | 262.586 | 16.205 | 0.014644 | 12.10 |
| 56 | adaptive | 373.215 | 19.319 | 0.032413 | 18.00 |
| 56 | communication_disabled | 373.215 | 19.319 | 0.032413 | 18.00 |

## Variance change: 200 versus 500 held-out episodes

Each seed's episode variance is a sample variance (denominators 199 and 499). This table first averages those variances equally across the 15 seeds, then reports 100 × (variance500 − variance200) / variance200. It is not a change in standard error or a mean of seed-specific percentage changes. Coverage variance uses fractions on the 0–1 scale; task-return variance uses squared return units. Individual seed changes can have either sign. More evaluation episodes improve the estimate; they do not inherently reduce outcome variance.

| Group | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |
| --- | --- | --- | --- | --- |
| adaptive | task_return | 296.709703 | 317.526628 | +7.02% |
| adaptive | coverage | 0.023385 | 0.021430 | -8.36% |
| communication_disabled | task_return | 299.579645 | 322.040850 | +7.50% |
| communication_disabled | coverage | 0.022889 | 0.021792 | -4.79% |
