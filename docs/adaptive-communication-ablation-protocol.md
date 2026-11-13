# Adaptive communication ablation: 500 held-out episodes per seed

Evaluate each of the 15 trained adaptive policies (training seeds 42-56) both
normally and with communication forcibly disabled, using the same **500
held-out scenarios (400000-400499)** as the expanded baseline. This is a frozen
policy evaluation intervention; there is no additional ablation training.

Both conditions use the exact same selected adaptive checkpoint for a seed.
The disabled condition preserves the movement chosen by the actor's original
deterministic ten-action joint argmax and replaces its transmit gate with zero.
It does not renormalize movement probabilities, replace the actor, substitute
a separately trained none policy, or force the original movement trajectory.
Recurrent state is reset at each episode and evolves normally from the
counterfactual observations. Removing messages may therefore change later
movement decisions.

Environment: three agents and three static landmarks; minimum pairwise radial
agent-start and landmark separation **0.4 units**; local sensing radius 0.6;
coverage radius 0.1; zero communication loss; horizon 100; collisions recorded
without terminating episodes. Use the original frozen whole-layout sampler,
28-input position/velocity communication protocol and `legacy-distance-v0`
task return. The exploratory landmark/coverage-sharing protocol is excluded.

Retain all original 200 normal and disabled episodes (400000-400199) without
altering outcomes. Add 300 scenarios (400200-400499) under each condition for
each seed. Normal adaptive results are shared with the adaptive study and
warm-none comparison. There are 7,500 normal and 7,500 disabled episodes;
only the 9,000 added normal-plus-disabled episodes need new evaluation.
Checkpoint hashes, source hashes, original episode hashes, scenario IDs and
starting-distance diagnostics are audited. The frozen evaluator also
reproduces an original scenario for each seed and condition. Every normal
trajectory with zero transmissions must match the disabled trajectory exactly.

Compare success, coverage, collision step rate, collision-episode fraction,
task return and attempted/delivered packets. Success means all landmarks are
covered simultaneously at the final step. Task return excludes packet penalties
and training bonuses. Differences are **disabled minus original**, computed
within each training seed, then summarized with equal seed weights and 95%
percentile bootstrap intervals (20,000 resamples; analysis RNG seed 20261110).
Report success-to-failure and failure-to-success episode counts separately.

Within-seed task-return and final-coverage variances use sample denominators
199 (200 episodes) and 499 (500 episodes). Report both variance and SD; coverage
variance uses fractions on the 0-1 scale and coverage SD uses percentage points.
The percentage-change summary first averages episode variances across the 15
seeds, then computes `100 * (variance_500 - variance_200) / variance_200`.
Across-seed variance of the 15 means is a separate statistic. See the
[variance comparison](scheduler-holdout-500-variance-comparison.md).

Run or resume all three related frozen-policy extensions with:

```powershell
.\.venv\Scripts\python.exe scripts/extend_scheduler_holdout.py --workers 6
```

Current results: [adaptive communication ablation report](adaptive-communication-ablation-results.md).
Expanded artifacts: `results/scheduler-heldout500-20261112/ablation/`.
Shared raw evaluations, checkpoint/source manifests and progress:
`results/scheduler-heldout500-20261112/`.
The original 200-episode ablation remains archived at
`results/adaptive-communication-ablation-20261112/`.
