# Larger-team policy tests at zero communication loss

This study evaluates the existing frozen adaptive, full, fixed_5 and none actors
in the same PettingZoo-compatible MPE2 SimpleSpread adapter used by the baseline.
Training-seed identifiers are 42-56. For every trained policy seed, 200 scenario
seeds (400000-400199) are evaluated at 5 agents/5 landmarks, 10/10 and 20/20.
The complete study contains 36,000 simulated episodes and 180 independent
training-seed/policy/team-size conditions. No larger-team policies are trained.

Environment rules remain zero packet loss, 100-step horizon, static landmarks,
sensing radius 0.6, coverage radius 0.1, pairwise radial agent-start center
distance >=0.4, and pairwise radial landmark center distance >=0.4. Initial
positions stay within the original [-1,1] square. The spacing rules are separate
for the agent set and landmark set; no new agent-to-landmark separation rule is
introduced. Agent spacing is a reset constraint, not a restriction on movement.
Collisions do not end episodes; success requires full simultaneous coverage
at the final step. All original scenario seeds are retained.

Dense N=20 layouts make whole-configuration rejection inefficient. All three
requested larger team sizes therefore explicitly use random sequential
placement with rejection against previously placed centers, whole-layout
restarts on exhaustion, and a final random permutation of entity labels. The
sampler is bounded, seeded, and never reduces spacing or enlarges bounds.
This changes the reset distribution relative to the original N=3 baseline.
Cross-size comparisons therefore measure transfer to these denser environments,
not a controlled isolation of agent count under the original reset distribution.
All 600 initial layouts are precomputed and hashed in layouts.json. Every policy
and training seed must reproduce the matching layout hash on every reset.

The decentralized actor has a fixed 28-value local observation and shares weights
across agents. It can infer actions for N=5/10/20 without changing or padding its
learned weights. Local entity slots remain limited to three neighbors/landmarks;
the actor does not receive global landmark coordinates. Recurrent memory is reset
for each episode and has an independent hidden state for each agent. The N=3
centralized critic is not used during evaluation.

The frozen baseline source is copied and hashed. Only layout.py, env.py and
evaluate.py gain explicit larger-team evaluation support; historical N=3 defaults
and whole-layout sampling remain unchanged. Baseline legacy-distance-v0 task
reward is retained in the evaluation snapshot: each step contributes
`-0.5 * sum(nearest-agent distance to each landmark) - collision_pairs / N`.
Task-return and task-return-per-agent means are both saved; raw returns naturally
scale with team size and should not be compared as equal-difficulty scores.

Success, final coverage, task return, collision step rate, collision-episode
fraction, overlapping pairs per step, attempted/delivered packets, and fraction
of full traffic are recorded per scenario and training seed. Zero loss requires
attempts to equal deliveries. Full traffic is N*(N-1)*100: 2,000, 9,000 and 38,000
packets per episode for the three team sizes. Fixed_5 sends 400, 1,800 and 7,600;
none sends zero. The adaptive policy uses its learned gates without a hard cap.

Across-seed results weight all 15 independent policy seeds equally and include
sample SDs and 95% percentile seed-bootstrap intervals (20,000 resamples).
Per-seed success intervals are Wilson episode intervals. Physical layouts are
paired across policies and model seeds within each N; they are not identical
across different N.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_team_sizes.py --workers 12
```

Outputs: `results/scheduler-team-sizes-zero-loss-20261111/`. See the
[results report](scheduler-team-size-zero-loss-results.md) and
[live local tracker](http://127.0.0.1:8765/teams/). The tracker also opens directly
from the study's progress.html and refreshes every five seconds. Per-case records
are in n-N/seed-S/episodes.jsonl; checkpoint hashes and full settings are in
manifest.json. The smoke test is separate and explicitly labeled.
