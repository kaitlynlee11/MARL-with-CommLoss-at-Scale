# Research protocol and implementation milestones

Kaitlyn Lee — EN.665.645 — proposal dated September 13, 2026.

## Refined question

Under partial observability, can a MAPPO-trained binary transmit policy maintain
cooperative landmark coverage while using fewer attempted transmissions than
continuous communication, and outperform periodic communication at comparable
traffic budgets as team size and independent packet loss increase?

[MAPPO](https://arxiv.org/abs/2103.01955) is a PPO approach with centralized value
training and decentralized policies. Communication decisions must be added to
its policy and objective. Particle navigation is a simulation proxy for robotics;
this project will not establish performance on physical robots.

## Testable hypotheses

1. At zero packet loss, adaptive scheduling reaches at least 95% of the full
   communication success rate while reducing attempted link transmissions by
   at least 30%. Treat those thresholds as proposed targets, fixed before final
   testing. If full communication barely succeeds, this relative criterion alone
   is insufficient: also report absolute success and coverage.
2. At comparable attempted traffic, adaptive scheduling achieves a higher task
   success rate than periodic scheduling. A positive difference must be supported
   by uncertainty across independent training seeds, not just evaluation episodes.
3. Report how both differences change with N and packet loss. Improvement at larger
   N or loss is a hypothesis to test, not an assumed property of the method.

Scope the main contribution to scheduling robustness with a small team-size
stress test. A fixed observation size alone is not evidence of scalable training,
networking, or generalization.

## Initial experimental contract

| Choice | Pilot setting |
| --- | --- |
| Task | N agents and N landmarks, MPE2 Simple Spread physics |
| Horizon | 100 simultaneous steps, no early success termination |
| Team sizes | 3, 5, 8; extend to 12 only after the pilot is stable |
| Loss | 0, 0.1, 0.3; 0.5 as a later stress condition |
| Sensing | Own position/velocity, three nearest visible peers and landmarks, radius 0.6 |
| Message | Fixed sender position/velocity, four float32 values |
| Network | All-to-all unicast, independent per-link loss, one-step delivery, TTL 5 |
| Reward | Mean MPE task reward minus lambda times attempted links per agent |
| Cost sweep | lambda = 0, 0.001, 0.01, 0.05, selected on validation seeds |
| Coverage | Landmark has any agent within 0.1 units |

The first implementation uses an absolute-coordinate mean inbox. It can hide
which agent occupied which location and may be insufficient for coordination.
Before large runs, compare it to a masked per-sender set encoder with position,
velocity, age, and validity. Pooling/attention should preserve fixed policy width
without pretending all-to-all communication becomes constant cost.

Holding arena size and physical radii fixed increases density when N increases.
This is a useful stress test, but confounds density with team size. Add a separate
density-controlled experiment (arena side proportional to sqrt(N), with sensor
range and movement units documented) before making a broad scalability claim.
The starter adapter implements the fixed-arena condition only.

## Baselines and fair comparisons

Train movement policies separately for no communication, communication every
step, periodic gates every k steps (k in 2, 5, 10), and learned gates. Keep actor
backbone, observation schema, horizon, environment steps, and evaluation seeds
consistent. Fixed baselines cannot improve by sending unless their movement
policies actually consume messages.

Add a Bernoulli random schedule matched to adaptive traffic, and later a simple
event rule such as transmit when velocity changes or a peer enters sensor range.
Match baseline traffic using validation runs; never tune on held-out test results.
Show success versus attempted bytes across operating points rather than comparing
one arbitrary lambda against one arbitrary period. Mean matched traffic does not
imply identical bursts; report that difference if it matters.

Use a full-observation diagnostic to see whether communication helps only because
of missing information. Also evaluate a policy with inbox contents zeroed at test
time; if results do not change, inspect whether the policy uses communication.

## MAPPO implementation plan

Use a shared actor with two categorical heads: movement (5 choices) and transmit
(2 choices). The joint action log probability is the sum of both head log
probabilities; PPO ratios must include the learned gate. Fixed baselines train
movement only and override the gate externally. Keep the same feature backbone.
Start with a feedforward policy using cached messages; recurrence can be a later
ablation if hidden-state memory is necessary.

Train a centralized critic with privileged world information while excluding it
from actor inputs. Start with separate fixed-N runs; concatenated world state is
acceptable there. For train-small/test-large transfer, use a masked set encoder
for critic entities and a shared actor that works across N. Shared parameters and
fixed actor dimensions make transfer possible but do not establish that it works.

Use horizon truncation as a time limit: bootstrap value targets at truncation,
not at genuine terminal states, and do not carry GAE across resets. Normalize
advantages, clip PPO updates and gradients, and log movement/gate entropy,
approximate KL, value loss, and explained variance. Track environment steps and
agent transitions separately since one environment step contains N actions.

First train N=3, loss=0 with each baseline and verify navigation learns. Then add
the gate and sweep costs. Use equal simulator-step budgets, with total agent
transitions, elapsed time and hardware reported. Choose the budget from pilot
learning curves before freezing the final protocol.

## Metrics and uncertainty

| Metric | Operational definition |
| --- | --- |
| Primary success | All landmarks covered simultaneously at the final horizon step |
| Coverage | Fraction of landmarks covered at the final step |
| Time to coverage | First step with simultaneous full coverage; report failures as censored |
| Collision step rate | Steps containing at least one overlapping unordered agent pair / episode steps |
| Collision pair rate | Overlapping unordered pair-steps / (steps times N(N-1)/2) |
| Task return | Sum of mean per-agent MPE task rewards, before communication penalty |
| Penalized return | Sum of shared training rewards after communication penalty |
| Traffic | Sender broadcasts, attempted directed packets, delivered packets, attempted payload bytes |
| Payload rate | Attempted payload bytes / simulator step; convert to bytes/s only if step duration is defined |
| Convergence | Simulator steps to a prespecified validation success threshold, plus evaluation-curve area |
| Scalability | Success, coverage, traffic per agent, wall time and memory versus N |

The first-success metric does not guarantee coverage is maintained; use final
success as the primary endpoint. Coverage permits one agent to count for nearby
landmarks; if unique assignment is required, change to bipartite matching before
final experiments. Collision diagnostics count each pair once; stock reward
penalties need not use that same counting convention.

Begin with three independent training seeds for pilots. For final claims, target
at least five training seeds and 100 held-out evaluation episodes per seed and
condition, subject to measured compute. Keep training, validation, and test seed
lists separate. Aggregate episodes within each trained run, then report mean and
95% intervals over training seeds (or hierarchical bootstrap). Episodes from a
single checkpoint are not independent training replicates. Pair scenario seeds
across methods for variance reduction, while documenting stochastic action and
loss handling. Report inconclusive intervals and failed training runs.

Train on clean communication first and test across losses; then compare against
training with randomized loss. Separately test larger teams without retraining
and policies retrained at each N. These answer different questions.

## Milestones and completion criteria

1. **Infrastructure (this scaffold):** test local sensing, seeded loss, stale
   messages, cost accounting, success/collision definitions and Parallel API;
   run random smoke episodes with metadata. No scientific performance claim.
2. **Navigation learning:** implement shared actor and centralized critic, PPO
   rollout/GAE updates, checkpoints and evaluation; demonstrate learned N=3
   navigation beats random movement on held-out seeds.
3. **Scheduling learning:** add binary gate to PPO, train fixed/no/full/adaptive
   policies, verify inbox ablation changes coordination, and trace when gates fire.
4. **Pilot and freeze:** select budgets, costs, baseline periods and success
   thresholds using validation only; save an explicit final experiment manifest.
5. **Study:** run seed replicates, loss/team-size sweeps and budget matching;
   plot task-success/traffic Pareto curves with uncertainty and write limitations.

The immediate next implementation task is milestone 2. Do not launch a large
sweep until trained navigation and communication usefulness are established.

## Reading and citation follow-up

Verified infrastructure references: [MPE2 Simple Spread documentation](https://mpe2.farama.org/environments/simple_spread/),
[PettingZoo MPE migration notice](https://pettingzoo.farama.org/main/environments/mpe/),
and [Yu et al., The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955).

The proposal's Foerster, Kim, Singh, Terry, Zhao and Zhu references remain a
reading list, not verified evidence of a research gap in this scaffold. Check
their original papers, author lists and publication details before submission.
Build a literature table with task, team sizes, topology, scheduling policy,
message loss model, traffic accounting and code availability. Frame the gap
around the specific conditions the experiments test rather than asserting that
scalability or robustness research is generally absent.
