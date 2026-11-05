# Task return at full coverage

Effective November 5, 2026, task reward uses version `coverage-gated-distance-v1`:

```text
task_reward = -0.5 * sum(nearest_agent_distance_to_each_landmark) * (not full_coverage)
              - overlapping_agent_pairs / number_of_agents
task_return = sum(task_reward over executed steps)
```

Full coverage means every landmark has an agent within 0.1 units simultaneously. Agents need not sit at the exact centers. A collision-free, fully covered step contributes zero task reward, so cumulative task return stays constant while coverage is maintained. Partial coverage retains the previous dense distance cost. Losing coverage resumes distance costs; collisions always retain their penalty, including a collision in a fully covered state.

This does not latch success at first coverage or end an episode early. In the collision-terminated experiments, any collision still terminates the episode as a failure. Success still requires collision-free survival through the scheduled 100 steps and full coverage at the final step. Physics, observations, 0.4-unit reset spacing, loss settings and the communication denominator of 600 packets are unchanged.

Task return excludes communication cost. Environment rewards still subtract `comm_cost * attempted_links / N`, including lost packets. Adaptive training retains its coverage/full-coverage bonuses, packet charge, collision and failure penalties, and the paper-inspired proximity cost. Its distance shaping cost now also vanishes at full coverage; its unfinished-step cost already did. These training rewards are separate from the reported task return.

The original adapter and vector training environment share the same task-reward function. New evaluation records, smoke records, replay data and training configurations identify the reward version. Existing checkpoints were not retrained, and original experiment JSONL files and aggregate reports retain their historical scores. Comparing task returns across these reward versions requires recalculation; success, collisions, steps and packet accounting do not depend on the scoring change for a frozen policy.

The paper-guided matched/fresh comparison replays and the [successful fresh-test replay](../results/paper-guided-adaptive-20261104/fresh/proposed-success-seed600001.html) were recalculated from their saved trajectories. Their original HTML files are preserved alongside them with `.legacy-distance.html` suffixes, and `.task-return-update.json` files record original hashes and old/new returns. The successful replay's existing `.json` retains the original source episode record; the revision sidecar identifies the current score.

For successful fresh seed 600001, full coverage is first reached at step 46 and maintained through step 100. Revised task return stays at **-22.0563834968** throughout that interval, rather than decreasing to the historical **-27.1179363455**. It still completes 100 steps with zero collisions and 14 attempts (14/600, or 2.33%). This is a scoring correction on a frozen policy, not an improvement measured after retraining.

To update an older replay while preserving its original HTML:

```powershell
.\.venv\Scripts\python.exe scripts/update_replay_task_return.py path/to/replay.html
```

The updater verifies the historical reward accumulation before recalculating, changes only the task-return values in the saved frames, and refuses to overwrite an existing archive. `scripts/make_success_replay.py --output path/to/new.html` generates a fresh replay with the current scoring and checks its trajectory statistics against the source evaluation record, allowing for the explicitly recorded reward-version difference.
