# Adaptive scheduler across 15 seeds

Training used `scripts/run_adaptive_seeds.py`, with archived outputs under
`results/adaptive-zero-loss-15-seeds-20261110/`. The archived auto-refreshing
`progress.html` opens directly from disk or through the local dashboard at
<http://127.0.0.1:8765/adaptive/>. Results update in
[the adaptive report](adaptive-15-seeds-results.md).

The current held-out evaluation has **500 episodes per training seed**, matching
the expanded baseline. Frozen-policy evaluation is extended with
`scripts/extend_scheduler_holdout.py`; current artifacts are under
`results/scheduler-heldout500-20261112/adaptive/`, with shared source/checkpoint
hashes and progress in its parent directory. The original 200-episode results
remain archived with the training study. Training and checkpoint selection are
unchanged.

Training seeds are 42-56. Each adaptive actor starts from the full-communication
checkpoint of the same corrected baseline seed. This transfers navigation
weights; the critic and optimizer are fresh. Joint movement and transmit actions
continue learning. Each initializer required 8,960,000 world steps of baseline
training; adaptive training may add up to 100,000,000 world steps per seed.
The budgets are recorded separately, so these results do not establish an
equal-compute advantage over the fixed baselines.

Training, validation, confirmation and held-out evaluation all use zero packet
loss, minimum pairwise radial agent-start separation 0.4, minimum pairwise radial
landmark separation 0.4, three agents, three static landmarks and a 100-step
horizon. Actors retain local sensing and recurrent memory. Collisions do not
terminate episodes, matching the corrected baseline workflow. Success means
simultaneous full landmark coverage at the final step. Collision step rate is
the fraction of the 100 steps with any overlapping agent pair.

The shared task-return implementation and simulator are copied, with verified
hashes, from the corrected baseline study's frozen source. Task-return scoring
therefore remains `legacy-distance-v0`. Adaptive training uses the existing
coverage/full-coverage bonuses, collision penalty, per-packet cost, unfinished
step penalty, coverage-gated distance shaping and projected dual packet penalty.
The stochastic training budget is 18% of full traffic; the deterministic target
is strictly below 20%, or fewer than 120 attempted directed packets per episode
on average. This is not a hard cap on individual episodes. Reports include the
maximum packet count and number of episodes at or above 120.

The existing trainer uses LR 0.0001, three PPO epochs, entropy 0.002, 64-unit
recurrent actors, 64 parallel worlds, 100-step rollouts, 25-step sequences and
one CPU thread per worker. Five seed workers run concurrently by default.

Every 25 updates, plus the first and last update, 200 batched validation worlds
from seed 100000 measure success and traffic. When success is strictly greater
than 90% and average packets strictly below 120, 200 separate original-environment
confirmation episodes (500000-500199) check the same targets. A confirmed pass
stops training and freezes `target.pt`; budget exhaustion freezes the best
validation-selected checkpoint instead. A `training/STOP` file requests a user
stop; that seed is not silently reported as completed.

Final evaluation uses 500 original-environment held-out scenarios (400000-400499)
per training seed, paired with all three baseline schedules. Held-out outcomes
never select checkpoints, alter hyperparameters or trigger training extensions.
A confirmation pass may fail the held-out targets. Budget-exhausted policies are
still evaluated and included. Failed processes are marked explicitly.

Individual results include success with Wilson episode intervals, coverage,
task return, packets and collision step rate. Across-seed summaries give equal
weight to each independent training seed and report sample SDs and 95% seed
bootstrap intervals. Paired per-seed changes compare adaptive outcomes against
full, none and fixed_5 on matching scenarios. Aggregate target attainment and
the number of individual seeds meeting both held-out targets are separate fields.

The original 200 scenarios (400000-400199) are retained without alteration;
300 additional scenarios (400200-400499) use the same frozen policy and simulator.
Each of the 15 seeds therefore has 500 held-out episodes, or 7,500 adaptive
episodes in total. Validation and confirmation still use 200 episodes each;
their scenarios and original stopping decisions are unchanged. The normal
adaptive outcomes are reused in the paired communication ablation and the
warm-start no-communication comparison.

Within-seed task-return and final-coverage variances use sample denominators
199 for the original 200 episodes and 499 for all 500. Coverage SD is reported
in percentage points. The variance-change summary first averages within-seed
variances equally across seeds, then computes
`100 * (variance_500 - variance_200) / variance_200`. Across-seed variance of
the 15 means is reported separately. See the
[variance comparison](scheduler-holdout-500-variance-comparison.md).

```powershell
.\.venv\Scripts\python.exe scripts/run_adaptive_seeds.py
```

Use `--smoke --output results/adaptive-seeds-smoke-20261110` for a separate short
pipeline test. Smoke results are explicitly labeled and never overwrite the
production report.

To reproduce the current held-out extension without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/extend_scheduler_holdout.py --workers 6
```

This command preserves completed episode records and resumes missing evaluations.
The training launcher above describes the archived training workflow; it is not
needed to extend held-out evaluation.
