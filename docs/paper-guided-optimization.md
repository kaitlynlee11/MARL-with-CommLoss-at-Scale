# Paper-guided adaptive communication optimization

This experiment uses the two supplied PDFs as research references and tests an adaptation in the existing repository environment. It does not reproduce either paper end to end or inherit either paper's theoretical guarantees.

## Sources and what transfers

1. **Scalable Cooperative Multi-Agent Reinforcement Learning with Adaptive Communication Range**, supplied anonymous ICLR 2026 review copy ([PDF](../../23951_Scalable_Cooperative_Mul.pdf)). Sections 3.1-3.3, Equations 23-27, Algorithm 1/Appendix C.7, Appendix D.1 and Table 1 on page 26 motivate separate communication/behavior policies, cost advantages with Lagrangian optimization, KL-limited updates, and pre-collision costs. Its actual control variable is communication range, with graph/locality assumptions and a different communication-cost model. Those assumptions and its monotonic-improvement result are not claimed here. Its CACC appendix uses absorbing collisions and a training penalty for potential collisions; our proximity penalty is a new adaptation of that idea to particle distances.
2. **Communication scheduling via proposed multi-agent reinforcement learning with centralized learning and distributed execution**, Ajay Nagendra Nama, Applied Intelligence (2026), DOI 10.1007/s10489-026-07202-4 ([PDF](../../s10489-026-07202-4.pdf)). Sections 3.1-4, Figures 1-2, and Table 2 describe separate action selection, scheduling weights, message encoding and centralized critics with communication lagging. The repository already has delayed delivery and centralized training. We preserve those features and adopt modular scheduling/behavior optimization. We do not introduce global Top(q) ranking, a learned message encoder, or Softmaxtanh: those would change the current execution/communication contract. The paper's execution-time results do not establish collision avoidance or landmark coverage in this environment.

Neither paper certifies that our proposed binary scheduler will avoid collisions. The risk critic/constraint, clock-augmented critic and finite-horizon return treatment below are explicit engineering extensions for this task.

## Environment held fixed

Three agents and three static landmarks; at least 0.4-unit pairwise radial agent starting separation and landmark separation; sensing radius 0.6; coverage radius 0.1; a 100-step scheduled horizon; zero packet loss; collision terminates an episode. Success requires surviving without collision through step 100 and simultaneous full landmark coverage at that final step. Observations remain the original 28 local values. Messages remain four float32 position/velocity values, 16 bytes per directed attempt, with one-step delivery and five-step cache expiration.

All agents execute only their own observation/memory-based policies. No action shielding, global-state execution, extra sensing, reset filtering or changes to physics are used. Both packet attempts/600 per scheduled episode and attempts per active step are reported, along with collision-episode fraction, coverage success and steps to termination.

## Proposed optimizer

The prior actor used one recurrent network and a ten-way joint categorical distribution over movement and transmit. The new actor has independent recurrent scheduler and behavior parameter blocks, with the same local input. Its law is factored as:

```text
p(transmit, movement | local history)
  = p(transmit | scheduler history)
    * p(movement | behavior history, transmit)
```

The behavior network has separate five-movement logits for each binary gate. The scheduler's two-way marginal is obtained from a copied ten-way head, allowing both recurrent networks to initialize directly from the original actor. At initialization the unnormalized joint logits, stochastic policy and greedy actions match the original checkpoint. This prevents an arbitrary reinitialization from being mistaken for an optimization benefit.

Training alternates scheduler updates with behavior updates, using independent Adam optimizers. The behavior stage uses the product of the updated scheduler importance ratio and the conditional behavior ratio, with scheduler gradients detached. PPO clipping is 0.1. An exact categorical KL target of 0.0075 stops further epochs when exceeded. This is an early-stop guard, not a hard guarantee that the last optimizer step stays inside the target. Actor LR is 0.0001; critic LR is 0.0003; two epochs and 25-step recurrent sequences are used. Gradient norms are clipped at 0.5.

There are separate centralized reward, packet-cost and collision-risk critics. An episode clock (elapsed steps / 100) is appended only to critic state, giving 37 critic inputs; actors still receive 28 local values. The reward critic transfers the prior critic with zero initial clock weight. Reward values use PopArt; cost critics are sigmoid-bounded because episode packet ratio and collision indicator lie in [0,1].

Reward and cost returns terminate at both collision and the scheduled finite-horizon end. This differs from the original trainer's time-limit value bootstrap. Episode costs use undiscounted GAE with gamma=1; reward GAE retains gamma=0.99. Reset boundaries stop all traces and independently clear each world's recurrent memories.

```text
actor advantage = reward advantage
                - packet_multiplier * packet-cost advantage
                - risk_multiplier * collision-cost advantage
```

Multipliers use projected dual ascent from completed episodes, not just active-step traffic:

- Packet episode cost: attempted packets / 600; training budget 0.18; multiplier starts at 50 and updates by 10 * (observed mean cost - 0.18).
- Collision episode cost: 1 on a fatal collision, otherwise 0; training risk budget 0.02; multiplier starts at 100 and updates by 20 * (observed collision fraction - 0.02).
- Both multipliers are projected into [0,1000]. These are experimental hyperparameters and soft constraints, not certified bounds.

The existing coverage reward, full-coverage bonus, fixed packet charge, timestep/distance penalties and finite collision-failure penalty remain. The proposed training reward additionally subtracts twice the following pre-collision proximity cost:

```text
sum over unordered agent pairs:
  clip((0.45 - center_distance) / (0.45 - 0.30), 0, 1)^2
```

This gives a warning signal before contact, is zero at separation >=0.45, and does not modify positions or action choices directly. It is a training-only objective change; evaluation uses the same collision and coverage definitions and unchanged base task returns.

## Controlled experiment

Both arms start from `results/collision-zero-loss-agent-spacing04-20261104/training/adaptive/best.pt`, whose selected lineage is 8,640,000 world steps. The original run's held-out result was 56% success and 23.5% collision episodes. Both follow-ups use training seed 42 and the same initial simulator-seed convention; each receives **3,200,000 additional world steps**. Rollout counts and active transitions, rather than wall-clock/FLOPs, are matched. The proposed method has more recurrent parameters and critics and therefore takes more optimizer computation per world step.

- **Baseline:** unchanged original checkpoint; no additional training.
- **Control:** joint PPO continuation, fresh Adam, actor/critic LR 0.0001, two epochs, 25-step sequences, full-rollout minibatches, original rewards, original active-step packet multiplier and original bootstrap treatment.
- **Proposed:** the modular optimizer and safety/cost changes above, fresh optimizers, and matched additional steps.

The proposed method is a package experiment; its result cannot attribute effects individually to architecture, KL control, cost critics, proximity shaping or horizon treatment. Component ablations would be needed for that.

Both arms use the same validation selection: first require packet ratio <0.20, then maximize success - 2 * collision-episode fraction; break ties by success then lower traffic. Validation uses 200 fixed batched worlds from seed 100000. The original checkpoint is included as a selection candidate in both arms. No held-out outcomes select a checkpoint or tune settings.

After both arms finish, frozen selected policies are evaluated on:

- **Matched set:** 200 original-environment scenarios, seeds 400000-400199, for comparison with prior reports.
- **Fresh set:** 200 original-environment scenarios, seeds 600000-600199, unused by the preceding experiments.

Each set includes baseline/control/proposed. Report success and collision fractions with Wilson episode intervals, packet ratios, attempts per active step and termination lengths. A safety improvement with lower coverage is reported as a tradeoff. One training seed and overlapping episode intervals do not establish broad superiority.

## Reproduce and monitor

```powershell
.venv\Scripts\python.exe scripts/run_paper_study.py --output results/paper-guided-repeat
```

[Live progress](../results/paper-guided-adaptive-20261104/progress.html). The run is complete; see [measured results](paper-guided-optimization-results.md), [full numerical results](../results/paper-guided-adaptive-20261104/report.md), [matched replay](../results/paper-guided-adaptive-20261104/matched/comparison.html), and [fresh replay](../results/paper-guided-adaptive-20261104/fresh/comparison.html). Per-update losses, cost multipliers and validation metrics are in `proposed/learning.jsonl` and `control/learning.jsonl`; manifest and selection files record source/PDF/checkpoint hashes and hyperparameters.

Verification before the full run: 56 regression tests passed, seven targeted modular-policy checks passed again after strengthening numerical equivalence, and a complete small two-arm training/evaluation/replay smoke pipeline completed. This verifies execution, not performance improvement. Source PDFs are unchanged.
