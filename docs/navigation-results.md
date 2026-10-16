# Three-agent MAPPO navigation results

This is the earlier **unconstrained-landmark, single-policy ablation** experiment.
The spaced-landmark rerun trains separate policies for each schedule. The training
commands below explicitly use `--min-landmark-distance 0` to reproduce this older
experiment; checkpoint evaluation infers the historical setting.

October 15, 2026. Navigation learning is implemented and trained. The retained
recurrent policy reached **81% final-step success and 90% coverage on validation**.
The table below uses 200 fresh held-out episodes per condition, seeds
300000–300199, in the original PettingZoo/MPE2 adapter. All learned-policy rows
use one frozen full-communication-trained checkpoint, with deterministic actions.

## Task and retained policy

Three agents, three landmarks, 100 simultaneous steps, sensing radius 0.6,
coverage radius 0.1, and independent packet loss fixed at zero in this comparison.
An episode succeeds only when all landmarks are covered simultaneously at the
final step. Agents do not know hidden landmark locations. Actor inputs remain
the original 28 values, including the pooled inbox; messages remain four float32
position/velocity values (16 payload bytes). The GRU remembers observation history.

The shared actor has a 64-unit encoder, a 64-unit GRU and five movement logits.
Communication gates are externally fixed, not learned. The centralized critic
has two 64-unit hidden layers, consumes physical state and pooled inboxes, and
uses PopArt value normalization. Critic inputs never enter the actor.

The retained training lineage contains 8,640,000 world steps (25,920,000 agent
transitions), seed 42, on CPU with two PyTorch threads. Exploratory pilot runs
are not included in that lineage count. A parent snapshot and SHA-256 identify
the exact 8,320,000-step checkpoint used for refinement.

Training reward was base task reward + one per covered landmark + an all-covered
bonus + assignment-distance potential shaping. The all-covered bonus was 5 in
the first phase and 10 in refinement. Potential scale was 5, with gamma 0.99 and
phi equal to negative optimal one-to-one assignment distance. The initial-state
curriculum mixed near-landmark starts with random starts for the first 1,000,000
steps only; all subsequent training and all evaluation used original random
starts. No demonstrations or hidden coordinates were supplied to the actor.

PPO used clip 0.2, GAE lambda 0.95 and gradient limit 0.5. The first phase used
25-step sequences, five epochs, minibatch 2048, learning rate 0.0005, entropy
coefficient 0.002 and joint gradient clipping. The final 320,000 steps used
100-step sequences, three epochs, minibatch 6400, learning rate 0.0003, entropy
0.001 and separate actor/critic clipping. Checkpoint selection used validation
only; the held-out comparison was run after freezing the checkpoint.

Reported returns exclude training bonuses, shaping and communication penalties.
Collision step rate is the fraction of steps with any overlapping agent pair.
Traffic counts attempted directed packets, including losses; here loss is zero.

## Held-out comparison

| Communication | Final success (95% episode interval) | Coverage | Task return | Attempted packets | Collision step rate |
| --- | --- | --- | --- | --- | --- |
| full | 76.5% (70.2%–81.8%) | 87.2% | -35.11 | 600 | 10.6% |
| fixed_2 | 75.0% (68.6%–80.5%) | 86.3% | -35.13 | 300 | 11.1% |
| fixed_5 | 78.0% (71.8%–83.2%) | 87.8% | -35.03 | 120 | 11.0% |
| fixed_10 | 68.0% (61.2%–74.1%) | 83.8% | -37.30 | 60 | 12.3% |
| none | 43.5% (36.8%–50.4%) | 71.0% | -41.74 | 0 | 13.2% |
| random_full | 0.0% (0.0%–1.9%) | 0.8% | -137.24 | 600 | 2.1% |

These are exploratory results for one trained policy, not a multi-seed study.
The Wilson intervals describe episode uncertainty, not training-seed variation.
Differences of a few percentage points should not be treated as established
improvements. The fixed/no-communication rows are test-time ablations, not policies
retrained for their communication conditions. The experiment does not yet test
adaptive scheduling, packet-loss robustness, or transfer to larger teams.

## Local artifacts and replay

- `results/mappo-full-gru-refine/best.pt`: selected policy and optimizer state.
- `results/mappo-full-gru-refine/parent.pt`: exact parent used in refinement.
- `results/mappo-full-gru-refine/config.json`: refinement parameters and parent hash.
- `results/mappo-full-gru-v1/learning.jsonl`: initial learning curve and validation.
- `results/mappo-full-gru-refine/learning.jsonl`: refinement learning curve.
- `results/navigation-evaluation/summary.json`: metrics, configuration and checkpoint hash.
- `results/navigation-evaluation/episodes.jsonl`: all 1,200 held-out episode records.
- `results/trained-success-demo.html`: first held-out full-success example, seed 300001.
- `results/trained-demo.html`: first held-out scenario, seed 300000; full communication
  reaches full coverage earlier but finishes with only two landmarks covered.

Both replays compare full/every-five-step/no communication on the same initial
world. Replay final metrics were checked against the evaluation records.
Artifacts under `results/` are local and ignored by Git; this note is durable.

## Reproduce

Install with `uv sync --extra dev --extra train`. Use new output directories.
Initial training (same configuration as the retained parent's training prefix):

```powershell
uv run --extra train marl-train --output results/repro-base --steps 8320000 --eval-every 50 --recurrent --hidden 64 --curriculum-steps 1000000 --entropy 0.002 --coverage-bonus 1 --success-bonus 5 --potential-scale 5 --gradient-clipping joint --min-landmark-distance 0
```

Refinement and evaluation:

```powershell
uv run --extra train marl-train --resume results/repro-base/best.pt --output results/repro-refine --steps 8640000 --eval-every 50 --recurrent --hidden 64 --lr 0.0003 --sequence-length 100 --minibatch 6400 --epochs 3 --entropy 0.001 --coverage-bonus 1 --success-bonus 10 --potential-scale 5 --gradient-clipping separate --min-landmark-distance 0
uv run --extra train marl-evaluate results/repro-refine/best.pt --episodes 200 --seed 300000 --output results/repro-evaluation
uv run --extra train marl-visualize --checkpoint results/repro-refine/best.pt --seed 300000 --loss 0 --output results/repro-demo.html
```

To evaluate the existing selected checkpoint, replace the checkpoint path with
`results/mappo-full-gru-refine/best.pt`; no retraining is needed. Environment and
optimizer RNG streams restart on continuation, as specified in the saved config.
Package versions are fixed in `uv.lock`; exact numerical reproduction can still
depend on platform and PyTorch kernels.
