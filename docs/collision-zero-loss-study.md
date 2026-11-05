# Collision-terminated scheduler comparison at zero packet loss

Latest recalculation: [results excluding reset-collision scenarios](collision-zero-loss-safe-start-results.md) uses the same 170 collision-free-start scenarios for every condition in both stages. The tables below preserve the original 200-scenario, all-start evaluation.

Completed November 4, 2026: existing frozen policies were evaluated first, then all three schedulers were independently retrained from scratch for 8,960,000 world steps each (26,880,000 total) and evaluated again. Both stages used 200 matched zero-loss episodes per condition. See the [combined report](../results/collision-zero-loss-20261104/report.md), [frozen report](../results/collision-zero-loss-20261104/frozen/report.md), and [retrained report](../results/collision-zero-loss-20261104/retrained/report.md).

## Environment and endpoints

The physical setup is unchanged: three agents and three static landmarks, minimum pairwise landmark center spacing 0.4, sensing radius 0.6, coverage radius 0.1, 100 scheduled steps, original MPE2 physics and 28-value local actor observations. Communication has zero packet loss. Payloads, one-step delivery and five-step cache expiration are unchanged.

An episode fails immediately upon any overlapping agent pair. Initial overlaps count as step-0 failures with no actions or communication; starts are not filtered in evaluation. Success requires both collision-free survival through step 100 and simultaneous full landmark coverage at that final step. Early full coverage does not end an episode. Safe but incomplete coverage at the horizon is also failure.

Each phase uses the same 200 original-environment seeds 400000-400199 for every condition. The frozen policies are the existing adaptive target checkpoint (trained at 25% loss) and the reproduced separately trained loss-free fixed_5/full checkpoints. This phase tests changed evaluation conditions without changing weights. It does not show how a policy learns collision avoidance.

## Frozen-policy results

| Scheduler | Collision-free success | Episodes with any collision | Attempted packets / 600 | Packets per episode | Attempts per active step | Mean executed steps |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Adaptive | 24.0% | 76.0% | 0.12% | 0.73 | 0.024 | 30.55 |
| Fixed every 5 steps | 26.5% | 72.0% | 7.21% | 43.26 | 1.239 | 34.91 |
| Full communication | 26.0% | 74.0% | 32.94% | 197.64 | 6.000 | 32.94 |

Fifteen percent of these scenarios already contain a collision at reset. Their inclusion imposes an 85% success ceiling on this evaluation set, even with perfect subsequent control. The earlier >90% criterion cannot be met on this set under the new failure rule without changing the reset distribution. This experiment retains the distribution and uses matched training budgets rather than that stopping criterion.

The full-policy traffic ratio is below 100% because episodes end early, not because its scheduler reduces its sending rate. It still attempts exactly six packets per active step.

## Collision-aware retraining

All three policies start from scratch with training seed 42, a 64-unit GRU actor, centralized critic and PopArt, one CPU PyTorch thread per condition, and equal world-step budgets. Their training environment also has zero packet loss. Three local Python training processes run concurrently.

Each receives 8,320,000 initial steps and 640,000 refinement steps (8,960,000 world steps total); early stopping is disabled. The initial phase uses five PPO epochs, learning rate 0.0005, entropy 0.002, 25-step sequences, minibatch 2048 and joint clipping. Refinement resumes the last initial checkpoint, retains weights/optimizer/packet multiplier, and uses three epochs, learning rate 0.0003, entropy 0.001, 100-step sequences, minibatch 6400 and separate clipping. Rollouts have 100 steps across 64 worlds. Resumed simulator/RNG streams restart as recorded in config.json. No near-landmark curriculum is used.

Each schedule uses the same team reward per active step:

```text
covered_landmark_count
+ 10 * simultaneous_full_coverage
- 5 * overlapping_agent_pair_count
- (0.05 + adaptive_packet_multiplier) * attempted_directed_packets
- 0.01 * (not simultaneous_full_coverage)
- 0.5 * sum_of_nearest_agent_distances_to_landmarks
- 100 * collision_terminal
```

The extra finite failure penalty is used instead of negative infinity so PPO advantages and gradients remain finite. The adaptive multiplier is zero for fixed/full policies. For adaptive transmission it follows projected dual ascent at rate 0.02 toward a stochastic active-step packet ratio of 18% (relative to six attempts per active step). This proxy is distinct from the reported fixed scheduled-episode denominator, and both metrics are logged/reported. The gate is learned without a forced transmit cap.

Collision terminations are true terminals with no critic bootstrap. Time-limit truncations retain bootstrapping. Only finished worlds reset, and recurrent memory resets independently for those worlds. At reset, an initially terminal world has no action to optimize: such failures are counted and immediately reset without inserting a synthetic action transition in PPO. Evaluation includes all initial failures and reports their fraction explicitly. This does not filter the evaluation distribution.

Validation uses 200 batched scenarios from seed 100000 at zero loss under the same failure rule. Checkpoints are selected by collision-free final success, then lower collision-episode fraction, coverage and lower packet ratio. Selection compares the best checkpoints from both phases. No held-out metrics select a checkpoint. Each selected policy then undergoes the same 200-scenario original-environment evaluation as the frozen phase.

## Collision-aware retrained results

| Scheduler | Success (95% Wilson) | Any-collision episodes | Attempted packets / 600 | Packets/episode | Attempts/active step | Mean executed steps |
| --- | --- | --- | --- | ---: | ---: | ---: |
| adaptive | 35.0% (28.7%-41.8%) | 20.5% | 1.71% | 10.28 | 0.126 | 81.51 |
| fixed_5 | 44.5% (37.8%-51.4%) | 16.0% | 16.85% | 101.10 | 1.200 | 84.22 |
| full | 52.0% (45.1%-58.8%) | 17.0% | 83.86% | 503.13 | 6.000 | 83.86 |

Changes relative to the frozen policies on the same scenarios:

- adaptive: success 24.0% -> 35.0% (+11.0 percentage points); collision episodes 76.0% -> 20.5%; attempts per active step 0.024 -> 0.126.
- fixed_5: success 26.5% -> 44.5% (+18.0 percentage points); collision episodes 72.0% -> 16.0%; attempts per active step 1.239 -> 1.200.
- full: success 26.0% -> 52.0% (+26.0 percentage points); collision episodes 74.0% -> 17.0%; attempts per active step 6.000 -> 6.000.

Traffic changes should be interpreted alongside survival time. Full communication always sends six packets per active step, so its larger post-training packet total reflects longer survival rather than a change in its schedule. The adaptive sending rate can change as well as survival time. Differences between the old and newly trained policies do not isolate communication as the cause of safety changes.

[Frozen comparison replay](../results/collision-zero-loss-20261104/frozen/comparison.html) | [Retrained comparison replay](../results/collision-zero-loss-20261104/retrained/comparison.html)

## Reported metrics

- Collision-free final full-coverage success and 95% Wilson episode interval.
- Fraction of episodes with any collision, including initial overlaps.
- Mean attempted packets divided by **600 packets per scheduled 100-step episode**, including early failures.
- Attempted packets per active step, defined as total attempts divided by total executed steps across all episodes. Step-0 failures contribute neither attempts nor steps.
- Mean and median executed steps before termination, min/max steps, horizon survival, and mean steps among collision failures.
- Initial-collision fraction and its implied success ceiling.

Task returns in evaluation are unchanged base environment returns accumulated only until termination; they exclude training bonuses/failure penalties and cannot be compared directly with historical 100-step returns. Single training seed per condition; episode intervals do not capture training-seed uncertainty.

## Reproduce and inspect

```powershell
.venv\Scripts\python.exe scripts/run_collision_study.py --output results/collision-zero-loss-repeat
```

This starts with frozen-policy evaluation if no frozen summary exists in the output, trains the three conditions with matched budgets, evaluates them, verifies replay endpoints against episode records, and writes comparison reports. The runner protects its manifest and training outputs against accidental reuse.

Artifacts under `results/collision-zero-loss-20261104/`:

- `frozen/`: completed original-policy evaluation, summary, episode records and replays.
- `training/{adaptive,fixed_5,full}/{initial,refine}/`: configuration, PPO logs and checkpoints.
- `training/{adaptive,fixed_5,full}/selection.json` and `best.pt`: validation selection and checkpoint hash.
- `retrained/`: new evaluation summaries, records and collision-terminated replays.
- `summary.json` and `report.md`: combined frozen/retrained comparison after completion.
- `status.json` and `progress.html`: current phase and auto-refreshing per-policy budget progress.

Checks: 40 regression tests passed. A complete separate smoke pipeline exercised all three initial/refinement paths, zero-loss original-environment evaluation and replay verification. The smoke run is execution verification, not research evidence. Historical and previous adaptive artifacts are preserved.
