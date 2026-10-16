# Packet-loss evaluation of spaced-landmark MAPPO policies

Frozen checkpoints from the loss-free study, tested with 25% independent packet loss per directed link per sending step.
Each separately trained policy keeps its own schedule; no additional training or checkpoint selection was performed.
Same 200 held-out seeds per condition, starting at 400000; three agents, 100 steps.
Nearby-only landmark sensing (radius 0.6); static landmarks with pairwise Euclidean center distance >= 0.4.

| Schedule | Success, no loss | Success, with loss (95% interval) | Change (pp) | Coverage | Task return | Attempted / delivered packets | Collision step rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 90.5% | 90.5% (85.6% to 93.8%) | +0.0 | 94.3% | -30.16 | 600 / 449.55 | 5.9% |
| none | 81.5% | 81.5% (75.5% to 86.3%) | +0.0 | 92.7% | -32.39 | 0 / 0.00 | 5.5% |
| fixed_5 | 91.5% | 91.5% (86.8% to 94.6%) | +0.0 | 95.8% | -28.45 | 120 / 90.19 | 5.2% |

Final success means all three landmarks are covered simultaneously at step 100 (coverage radius 0.1).
Minimum observed landmark separation: 0.403667.
Failed deliveries retain older cached data until its five-step expiration; no retransmission is added.
Packets carry 16 payload bytes; attempted bytes still count when delivery fails. Task return excludes communication penalties.
The no-communication trajectories and metrics match the loss-free baseline exactly.
One training seed per schedule. Wilson intervals describe episode uncertainty, not variation across training seeds.
This measures test-time robustness to loss, rather than performance after training with loss.

## Paired episode changes

| Schedule | Success becomes failure | Failure becomes success |
| --- | --- | --- |
| full | 11 | 11 |
| none | 0 | 0 |
| fixed_5 | 12 | 12 |

Aggregate success stayed unchanged because the success/failure changes balanced
within each communicating condition. Average task return worsened for both
communicating policies, and collision step rates increased. Full communication
coverage fell from 95.7% to 94.3%. This single-seed experiment does not establish
that packet loss has no effect.

## Delivery checks

- full: observed packet loss 25.08% (120,000 attempted packets).
- fixed_5: observed packet loss 24.84% (24,000 attempted packets).
- none: no attempted packets; observed packet loss is undefined.

## Reproduce

```powershell
uv run --extra train python -m limited_comm.loss_study --output results/spaced-navigation-loss25-repeat
```

Source checkpoints: `results/spaced-navigation/{full,none,fixed_5}/best.pt`.
Results: `results/spaced-navigation-loss25/summary.json` and `episodes.jsonl` (600 records).
`comparison.html` shows the first held-out seed, selected before observing outcomes.
`loss-affected.html`, if present, deliberately selects the first seed where a previously successful communication policy fails with loss.
Checkpoint hashes/configurations are recorded in summary.json; original artifacts are preserved.
