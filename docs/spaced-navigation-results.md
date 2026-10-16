# Spaced-landmark MAPPO baselines

The same frozen checkpoints were subsequently tested with 25% packet loss per
directed link. See [packet-loss results](spaced-navigation-loss25-results.md)
for the paired comparison and replays.

Three policies were trained independently from scratch, with identical two-phase budgets and hyperparameters.
Each policy was evaluated under its own training schedule on the same held-out scenario seeds.
Pairwise landmark center distance >= 0.4; loss=0; sensing radius=0.6; horizon=100.
Training work: 8,960,000 world steps per policy. Training seed: 42.
Held-out episodes per policy: 200; seeds 400000 through 400199.

| Training and evaluation schedule | Final success (95% episode interval) | Coverage | Task return | Packets/episode | Collision step rate |
| --- | --- | --- | --- | --- | --- |
| full | 90.5% (85.6% to 93.8%) | 95.7% | -29.41 | 600 | 5.5% |
| none | 81.5% (75.5% to 86.3%) | 92.7% | -32.39 | 0 | 5.5% |
| fixed_5 | 91.5% (86.8% to 94.6%) | 95.8% | -27.50 | 120 | 4.6% |

Success requires simultaneous coverage of all three landmarks at the final step (center distance <= 0.1).
All training, curriculum, validation, evaluation and replay layouts obey the same spacing rule; landmarks never move.
The smallest observed held-out landmark distance was 0.403667.
No hidden landmark coordinates are supplied to actors. Inputs remain the original 28 values; GRU memory retains past observations.
Messages contain position/velocity only (16 payload bytes). Cost is reported separately from the base task return.
Training rewards add one per covered landmark, an all-covered bonus of 5 initially/10 in refinement, and assignment-potential scale 5.
A near-landmark initial-state curriculum is annealed to original random agent starts by 1,000,000 steps.
All schedules use a 64-unit GRU. Phase 1: LR 0.0005, entropy 0.002, 5 epochs, 25-step sequences, joint clipping.
Phase 2: LR 0.0003, entropy 0.001, 3 epochs, 100-step sequences, separate clipping. Early stopping is disabled for equal budgets.
Final checkpoints are selected by validation success then coverage, comparing the best checkpoints from both phases.
This is one training seed per condition, not a multi-seed study. Intervals measure episode uncertainty only.
These results use a changed layout distribution and independently trained policies; they are not directly comparable to the older test-time ablations.

Local artifacts: each condition's best.pt, selection.json, initial/refine/config.json and learning.jsonl; root summary.json and episodes.jsonl.
comparison.html uses the first held-out seed; all-success.html, if present, selects the first held-out seed where all three policies succeeded.


## Reproduce and inspect

```powershell
uv sync --extra dev --extra train
uv run --extra train marl-study --output results/spaced-navigation-repeat
```

The suite trains all three schedules independently with one matched training seed
(42). The initial budget is 8,320,000 steps, followed by 640,000 refinement steps;
checkpoints are selected only by validation. Evaluation seeds are 400000–400199.
Individual checkpoint configurations preserve all PPO, reward and curriculum settings.

Open `results/spaced-navigation/comparison.html` for the first held-out scenario,
or `results/spaced-navigation/all-success.html` for the first held-out scenario
where all three policies succeeded. Each panel uses its own trained checkpoint.
Checkpoint files are `results/spaced-navigation/{full,none,fixed_5}/best.pt`.

Verification: 600 episode records; zero lost packets; paired layouts across all
three conditions; checkpoint hashes/configurations; every frame of the generated
replays respects the 0.4 separation and static-landmark requirements.
Artifacts under `results/` are local and ignored by Git; this report lives in
`docs/` so it can be versioned with the project. Historical unconstrained results
are preserved separately.

Every-five-step communication used 80% fewer directed packets than full
communication, with similar observed success. Its one-percentage-point advantage
does not establish a performance improvement. These are single-training-seed results.
