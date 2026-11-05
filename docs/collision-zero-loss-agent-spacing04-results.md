# Full collision-terminated experiments with 0.4-unit agent starts

Completed November 4, 2026. Both requested stages are complete: frozen-policy evaluation under the new reset distribution, followed by independent from-scratch retraining and reevaluation of adaptive, fixed-five-step and full-communication policies.

## Setup

- Three agents and three static landmarks; 100 scheduled steps; zero communication packet loss.
- All agent centers start at least 0.4 units apart radially. Entire invalid agent layouts are resampled uniformly in the original [-1,1] square. This is a reset constraint; agents may move closer later.
- Landmark center separation remains at least 0.4; local sensing radius remains 0.6; landmark coverage radius remains 0.1; overlapping agent centers closer than 0.3 constitute a collision.
- Any collision immediately terminates the episode. Success requires collision-free survival through step 100 and full landmark coverage at that final step. Early full coverage does not end the episode.
- Both stages use all 200 matched seeds 400000-400199 per scheduler. No scenarios are filtered or replaced after evaluation. There are no reset collisions and no initial-overlap-induced 85% ceiling.
- Packet ratio is mean attempts divided by 600 packets per scheduled 100-step episode. Attempts per active step is total attempts divided by total executed steps. Later collision failures can still reduce the scheduled-episode packet ratio.

## Results

| Stage | Scheduler | Collision-free success (95% Wilson) | Any collision | Packets / 600 | Packets/episode | Attempts/active step | Mean termination step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| frozen | adaptive | 27.0% (21.3%-33.5%) | 73.0% | 0.10% | 0.63 | 0.018 | 35.75 |
| frozen | fixed_5 | 32.5% (26.4%-39.3%) | 67.0% | 8.51% | 51.03 | 1.237 | 41.26 |
| frozen | full | 34.0% (27.8%-40.8%) | 65.5% | 42.35% | 254.13 | 6.000 | 42.35 |
| retrained | adaptive | 56.0% (49.1%-62.7%) | 23.5% | 6.04% | 36.21 | 0.434 | 83.50 |
| retrained | fixed_5 | 51.0% (44.1%-57.8%) | 23.5% | 16.67% | 100.02 | 1.205 | 82.98 |
| retrained | full | 36.5% (30.1%-43.4%) | 26.5% | 79.08% | 474.48 | 6.000 | 79.08 |

## Termination details

| Stage | Scheduler | Horizon survival | Median active steps | Min-max active steps | Mean steps among collision failures |
| --- | --- | ---: | ---: | --- | ---: |
| frozen | adaptive | 27.0% | 14.0 | 3-100 | 11.99 |
| frozen | fixed_5 | 33.0% | 15.0 | 3-100 | 12.33 |
| frozen | full | 34.5% | 15.0 | 3-100 | 11.99 |
| retrained | adaptive | 76.5% | 100.0 | 5-100 | 29.77 |
| retrained | fixed_5 | 76.5% | 100.0 | 5-100 | 27.57 |
| retrained | full | 73.5% | 100.0 | 5-100 | 21.06 |

## Training and interpretation

Each new policy completed 8,320,000 initial plus 640,000 refinement world steps: 8,960,000 per policy and 26,880,000 total. All use training seed 42, a 64-unit recurrent actor and centralized critic, 64 parallel worlds, one CPU PyTorch thread per training process, and matched PPO budgets. Early stopping is disabled. Selection compares the best validation checkpoints from both phases; held-out outcomes are not used for selection.
The initial phase uses LR 0.0005, entropy 0.002, five PPO epochs, 25-step sequences, minibatch 2048 and joint gradient clipping. Refinement uses LR 0.0003, entropy 0.001, three epochs, 100-step sequences, minibatch 6400 and separate clipping. Reset curriculum is disabled. Training and validation both enforce the new agent separation.
The shared team reward is covered-landmark count + 10 times full coverage - 5 times collision pairs - per-attempt packet cost - 0.01 per unfinished step - 0.5 times nearest-landmark distance sum - an extra finite 100-point terminal collision penalty. The fixed packet charge is 0.05; adaptive transmission additionally uses a nonnegative dual multiplier updated toward an 18% stochastic active-step traffic proxy. This proxy is distinct from the reported attempts/600 metric.
Collision terminals have zero value bootstrap; time-limit truncations bootstrap normally. Recurrent memory and physical state reset independently for finished worlds.
Frozen policies retain the original pretrained weights: the adaptive target from the 25%-loss study and the earlier zero-loss fixed/full policies. They are tested here at zero loss with the new start distribution. The retrained policies are new from-scratch policies under the new reset and collision rules. These are different training histories, not a controlled isolation of communication alone.
Previous safe-start filtered results excluded initial overlaps at the 0.3 collision threshold from an unrestricted-start experiment. The new 0.4 starting-distance distribution is stricter and resamples layouts; those prior 170-scenario results are not the same cohort as this new 200-scenario experiment.
One training seed per condition. Wilson intervals describe episode uncertainty, not variation across training seeds. The experiment uses matched finite budgets; a completed run does not itself establish >90% success or the adaptive traffic target.

## Artifacts and verification

- [Full run manifest](../results/collision-zero-loss-agent-spacing04-20261104/manifest.json)
- [Frozen and retrained summaries](../results/collision-zero-loss-agent-spacing04-20261104/summary.json)
- [Frozen report](../results/collision-zero-loss-agent-spacing04-20261104/frozen/report.md) and [retrained report](../results/collision-zero-loss-agent-spacing04-20261104/retrained/report.md)
- [Frozen replay](../results/collision-zero-loss-agent-spacing04-20261104/frozen/comparison.html) and [retrained replay](../results/collision-zero-loss-agent-spacing04-20261104/retrained/comparison.html)
- [Frozen episode records](../results/collision-zero-loss-agent-spacing04-20261104/frozen/episodes.jsonl) and [retrained episode records](../results/collision-zero-loss-agent-spacing04-20261104/retrained/episodes.jsonl)
- [Finished progress page](../results/collision-zero-loss-agent-spacing04-20261104/progress.html)

The implementation passed 49 tests and an end-to-end spacing smoke pipeline before this full run. Final checks verified 1,200 comparative evaluation records; identical scenario seeds and starting-distance diagnostics across conditions/stages; every initial agent distance >=0.4; zero reset collisions; completed training budgets; finite optimization metrics; checkpoint hashes; packet accounting; and replay endpoints. Historical result files are preserved.
