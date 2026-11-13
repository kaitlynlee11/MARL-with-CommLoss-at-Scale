# Warm-start no-communication controls with matched optimization

This control tests whether the adaptive policy's performance advantage over the
original no-communication baseline can be explained by transferred navigation,
additional optimization, and the adaptive navigation objective rather than
communication. There are 15 new training runs, using seeds 42-56.

The three comparison groups are:

1. **Initial none:** the original independently trained no-communication baseline,
   with 8,960,000 world steps and its original two-phase training procedure.
2. **Warm none:** a new five-action movement policy, initialized from the exact
   same full-communication actor checkpoint used by adaptive for that seed.
3. **Adaptive:** the previously trained joint movement/transmit policy, unchanged.

Warm none and adaptive have identical initializer hashes and inherited baseline
training work. Warm none matches each seed's **actual completed adaptive world
steps**, not the unused 100-million-step ceiling. PPO epochs, recurrent sequence
length, minibatch size, rollout length, world count, learning rate, entropy
coefficient, gradient clipping, GAE, discount, PPO clipping and PopArt are matched.
Actual optimizer-step counters in the final Adam state are checked against the
expected update budget. Early stopping is disabled for warm none.

To preserve the same freshly initialized critic, construction follows the
original adaptive RNG order: create the seeded ten-action model, load the same
full-communication initializer, then replace only the actor with its five-action
movement actor. The resulting movement probabilities equal the adaptive warm
start's movement marginal. The fresh critic weights match byte for byte; no
critic or optimizer state is transferred from the initializer. `initial.pt`
and `initializer.pt` preserve the states needed to audit this. The control has
five movement actions rather than extra dummy transmit choices.

The frozen original training source is copied into the new study. Guarded edits
to train.py permit the no-communication warm start, apply the adaptive navigation
reward without its zero-valued packet term, and preserve the initial state. The
simulator, observations, PPO update loop and shared task-return implementation
remain the original source. Every source version and modification is hashed.

The adaptive navigation reward is covered-landmark bonus 1, full-coverage bonus
10, collision-pair penalty 5, unfinished-step penalty 0.01, and coverage-gated
distance penalty 0.5. Every warm-none transmit gate is zero during both training
and evaluation, so the packet charge is zero. The original none baseline used a
different training procedure and has less total optimization work; it is a
reference, not an equal-compute control.

Primary evaluation uses **final last.pt** at the matched optimization step,
corresponding to adaptive's final target.pt step. The validation-best checkpoint
is preserved for diagnostics but does not select the reported warm-none policy.
Held-out outcomes never select checkpoints, extend budgets or tune settings.

All groups use the same 500 held-out scenarios (400000-400499) for each
training seed: three agents/landmarks, zero loss, minimum pairwise radial agent
and landmark separation 0.4, local sensing radius 0.6 and 100 steps without
collision termination. This uses the legacy 28-input communication protocol,
not the exploratory landmark-sharing extension. Initial-none and adaptive
evaluations are reused; every new control episode must match their scenario
seed and starting-distance diagnostics. Evaluation return is the legacy base
task reward, excluding packet penalties and training bonuses.

Results include success, coverage, collision step/episode rates, task return,
packets and work per seed. Summaries compare the same completed seed set across
all three groups; partial results are provisional. Across-seed means, SDs and
95% percentile bootstrap intervals use 20,000 resamples. Paired differences
compare warm none with initial none and adaptive on matching seeds.

The evaluation extension retains the original 200 episodes (400000-400199)
and adds 300 (400200-400499) per policy seed. All 15 warm-none checkpoints
remain the original final `last.pt`; no retraining or reselection occurs.
Initial none reuses the completed 500-episode baseline evaluation, and adaptive
reuses the same 500-episode outcomes as the adaptive and ablation studies.
Thus every group has 7,500 episodes, with identical scenario IDs and starting
spacing diagnostics. Validation sets, training budgets and optimizer counters
remain unchanged.

Within-seed task-return and final-coverage variances use the episode sample
variance (denominators 199 and 499). SD is its square root; coverage SD is in
percentage points. The policy-level percentage change is calculated after
averaging the 15 within-seed variances equally:
`100 * (mean_variance_500 - mean_variance_200) / mean_variance_200`.
Across-seed variance of the 15 means is distinct and reported separately in the
[variance comparison](scheduler-holdout-500-variance-comparison.md).

Original training used `scripts/train_warm_no_comm.py`; training artifacts and
the archived 200-episode results remain under
`results/warm-no-communication-15-seeds-20261112/`. Current 500-episode evaluation
artifacts are under `results/scheduler-heldout500-20261112/warm-none/`, with
shared checkpoint hashes, raw evaluations and progress in its parent directory.
Extend or resume the frozen-policy evaluation with:

```powershell
.\.venv\Scripts\python.exe scripts/extend_scheduler_holdout.py --workers 6
```

See
[the comparison report](warm-no-communication-results.md) and
the archived [training tracker](http://127.0.0.1:8765/warm-none/).
