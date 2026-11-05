# Adaptive communication scheduler training

November 5 reward update: [distance penalties now stop while full coverage is maintained](task-return.md), in both the base task reward and adaptive training shaping. The formulas and completed November 4 experiments below describe the earlier reward; those checkpoints and historical measurements have not been retrained or overwritten.

Started November 4, 2026. This extends recurrent MAPPO to learn movement and transmission jointly. Each decentralized actor receives only the original 28 local observation values and recurrent memory. Its ten categorical actions encode `movement + 5 * transmit`, covering five movements and two gate choices. The centralized critic alone receives privileged world state and pooled inboxes.

## Environment and targets

The setup matches the reproduced spaced-navigation baseline: three agents, three static landmarks, minimum pairwise landmark center distance 0.4, sensing radius 0.6, coverage radius 0.1, and a fixed 100-step horizon. Packet loss is known and fixed in the environment at 25%, independently per directed link. Its configured rate is not an extra actor observation; there are no acknowledgments or hidden landmark coordinates. Messages retain position/velocity payloads, one-step delivery and five-step cache expiration.

Full communication attempts `3 * 2 * 100 = 600` directed packets per episode, including dropped packets. The adaptive target is **strictly fewer than 120 attempts per episode on average**, or less than 20% of full communication. This is an average traffic target, not a per-episode cap. No hard transmission mask enforces the budget.

Training stops successfully only when deterministic evaluation has both:

- More than 90% final-step simultaneous full-landmark coverage.
- Mean attempted packet ratio strictly below 20% of the full baseline.

At 200 evaluation episodes, success requires at least 181 successful episodes. Exactly 90% success or exactly 20% traffic does not pass.

## Team reward per step

```text
reward = 1.0 * covered_landmark_count
       + 10.0 * simultaneous_full_coverage
       - 5.0 * overlapping_agent_pair_count
       - (0.05 + packet_multiplier) * attempted_directed_packet_count
       - 0.01 * (not simultaneous_full_coverage)
       - 0.5 * sum_of_nearest_agent_distances_to_landmarks
```

The nonnegative packet multiplier starts at zero and follows projected dual ascent after each PPO update:

```text
packet_multiplier = max(0, packet_multiplier + 0.02 * (sampled_packet_ratio - 0.18))
```

The ratio is measured over stochastic training attempts divided by full-communication attempts for that rollout. The 18% training budget supplies margin below the strict 20% deterministic evaluation limit. The multiplier increases above the budget and decreases below it, adding a learned constraint penalty to the fixed 0.05 charge per attempted packet. This encourages compliance without forcibly suppressing gates; success and traffic are still checked separately in evaluation. Effective penalties and sampled ratios are logged per update.

The full-coverage bonus is the largest positive component (10, compared with at most 3 for covered landmarks). Collisions cost 5 per overlapping pair per step; dropped packets incur the same attempt penalty as delivered packets. The small unfinished-step penalty encourages earlier coverage while maintaining the baseline's full horizon and final-step success definition. The distance term supplies a dense navigation signal. These coefficients are initial experimental choices, not tuned or validated optima. Task returns in evaluation remain the unchanged base environment return, separate from this training reward.

## Initialization and optimization

Movement is initialized from `results/reproduction-20261028/spaced-navigation/full/best.pt`. Encoder and GRU weights transfer; the movement head is duplicated into silent/transmit actions so the movement marginal is preserved. Initial stochastic transmit probability is 0.1, while deterministic inference initially prefers silence. The actor then learns both components; movement is not frozen. A new critic and optimizer are trained for the new reward. This is warm-start training, not an independent from-scratch baseline, and additional adaptive world steps are logged separately from initializer training.

Run settings: training seed 42; CPU with two PyTorch threads; hidden size 64; 64 parallel worlds; 100-step rollouts; 25-step recurrent sequences; three PPO epochs; minibatch 2048; learning rate 0.0001; entropy coefficient 0.002; separate gradient clipping at 0.5. GAE, gamma, PPO clipping and PopArt retain the existing trainer settings. There is no new near-landmark curriculum, potential shaping, or discovery bonus in this run.

Every 25 updates (160,000 world steps), plus the first and final update, deterministic batched validation evaluates 200 worlds using seed 100000. When both targets pass, the original PettingZoo/MPE2 adapter confirms them on seeds 500000-500199. These confirmation episodes participate in stopping and are not the final held-out set. A qualifying checkpoint is saved as `target.pt`. If confirmation fails, training continues. Validation and confirmation use the same 25% loss as training.

After training ends, the runner freezes the qualifying checkpoint (or the best available checkpoint on budget exhaustion) and evaluates 200 baseline-matched original-environment seeds 400000-400199. Final held-out results are reported independently; passing the stopping set does not guarantee passing the held-out targets. A self-contained adaptive HTML replay uses seed 400000.

## Run and monitor

From the repository root, with the locked training dependencies installed:

```powershell
.venv\Scripts\python.exe scripts/run_adaptive.py --output results/adaptive-loss25-20261104-constrained
```

The initial ceiling is 100,000,000 additional adaptive world steps. Training stops earlier when both targets are confirmed. Budget exhaustion is explicitly recorded as `budget_exhausted`, without claiming success. Output directories cannot be reused accidentally. An empty `STOP` file in the training directory saves `last.pt` and exits at the next PPO update boundary; the runner honors that stop without proceeding to evaluation.

Open `results/adaptive-loss25-20261104-constrained/progress.html` for auto-refreshing budget, success and packet-ratio bars. Budget progress is distinct from progress toward the success/traffic targets. Logs and artifacts:

- `training.log`: training and validation output.
- `training/learning.jsonl`: PPO and validation metrics, plus confirmation results when attempted.
- `training/config.json`: reward, initialization hash, environment and optimizer settings.
- `training/initializer.pt`: preserved navigation initializer.
- `training/best.pt`, `training/last.pt`: validation-selected and most recent checkpoints.
- `training/confirmation-<steps>.json`: original-environment confirmation episodes and summary.
- `training/target.pt`: created only when both confirmation targets pass.
- `status.json` and `training/status.json`: pipeline and trainer states.
- `evaluation/episodes.jsonl`, `evaluation/summary.json`, `summary.json`, `report.md`: final frozen-policy evaluation and target accounting.
- `comparison.html`: adaptive-policy replay, with actual learned transmit gates.

Implementation checks: 34 tests passed, including strict stop thresholds, attempted-packet charging at total loss, reward decomposition, recurrent joint-action log-probability replay and gradients, adaptive checkpoint roundtrip, original-environment action decoding and replay packet totals. A separate 12,800-step warm-start smoke run completed at 25% loss; that run establishes execution, not target attainment.

This is one training-seed experiment. Success thresholds use point estimates, not a requirement that the confidence-interval lower bound exceed 90%. Report Wilson episode intervals and collision metrics, and do not interpret reaching the stopping target as evidence across training seeds. Existing historical/reproduction artifacts remain separate.

An initial fixed-penalty pilot is preserved in `results/adaptive-loss25-20261104/`. Its early updates exceeded the traffic budget, prompting the dual-constraint version described here. The pilot was stopped through its STOP file before restarting from the same navigation initializer. It is not the constrained run or evidence of target attainment.
