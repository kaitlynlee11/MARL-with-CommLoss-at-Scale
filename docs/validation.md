# Scaffold verification

Verified October 5, 2026 on Windows with Python 3.12.13, MPE2 1.1.0,
PettingZoo 1.27.0, Gymnasium 1.3.0 and NumPy 2.5.3. Dependencies are recorded
in `uv.lock`.

`python -m pytest -q`: **11 passed**. MPE2's legacy environment factory emitted
two upstream deprecation warnings; the pinned version passed the Parallel API
test. Tests cover full/no delivery, charged failed packets, cache expiration,
message age, reproducibility, local observation isolation, reset behavior,
coverage/collision definitions and paired baseline accounting.

```powershell
python -m limited_comm.experiment --agents 3 5 8 --loss 0 0.1 0.3 --episodes 3 --horizon 100 --output results/first-smoke.jsonl
```

Completed **81 episodes**: 3 sizes × 3 losses × 3 schedules × 3 scenario seeds.
Local artifacts: `results/first-smoke.jsonl` and
`results/first-smoke.metadata.json`. Results are ignored by version control.

These runs validate execution and telemetry with random navigation. They do not
measure learning, convergence, adaptive communication or trained-policy quality.
These smoke results remain separate from learned-policy results below.

## MAPPO milestone — October 15, 2026

`python -m pytest -q`: **20 passed**, with the same two upstream environment
factory deprecation warnings. The added checks cover batched/original environment
parity under no/full/fixed communication, GAE truncation and termination semantics,
checkpoint roundtrips, PopArt prediction preservation, recurrent history/reset
behavior, recurrent log-probability replay and assignment-potential symmetry.

Training and evaluation use the optional `train` dependency group, including
PyTorch 2.14.1+cpu. The selected GRU policy reached 81% final-step success on 100
validation worlds. Its frozen checkpoint was evaluated on 200 fresh episodes per
communication condition in the original PettingZoo environment. See
[navigation-results.md](navigation-results.md) for the retained training protocol,
held-out metrics, uncertainty, checkpoint paths and replay examples.

## Spaced-landmark rerun — October 15, 2026

**24 tests passed.** Additional checks cover seeded rejection sampling, radial
distance, static landmarks across episode steps, schedule phase/counts, and
spacing in batched worlds. The three-policy suite was also exercised end to end
with a short training/evaluation run before the full study.

The full study trained three separate policies for 8,960,000 world steps each.
Its 600 held-out episode records were verified for minimum separation >= 0.4,
paired layouts, zero lost packets, and expected traffic. Both generated replays
were checked at every frame for static, adequately separated landmarks; replay
outcomes and returns matched the evaluation records. Checkpoint hashes and
schedule/radius/spacing configurations were also verified. Full results are in
[spaced-navigation-results.md](spaced-navigation-results.md).

## Packet-loss evaluation - October 15, 2026

**25 tests passed.** The added channel check compares partial-loss deliveries
against independent directed-link draws at a 0.25 threshold, including silent
steps and attempted-payload accounting.

The frozen spaced-landmark checkpoints were evaluated on the original 200 paired
held-out seeds per schedule at 25% packet loss. The evaluation asserts checkpoint
hashes, training schedules, sensing radius, horizon, minimum landmark separation,
matching attempted traffic, and exact no-communication metric equivalence with
the loss-free baseline. Replay outcomes, returns and delivery counts are matched
to episode records; every replay frame is checked for static landmarks with
pairwise radial distance >= 0.4. See
[spaced-navigation-loss25-results.md](spaced-navigation-loss25-results.md).
