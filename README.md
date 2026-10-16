# Adaptive communication scheduling for cooperative navigation

Starter project for Kaitlyn Lee's EN.665.645 research proposal. The goal is to test
whether learned communication timing improves the tradeoff between coordination
and communication cost as teams grow and messages are lost.

**Current milestone:** environment, channel, smoke runs, MAPPO navigation training,
checkpoint evaluation and replay. The transmit schedule is externally fixed;
a learned adaptive scheduler is not implemented yet. Smoke runs remain random.

The latest separately trained baselines and replays are documented in
[spaced-landmark results](docs/spaced-navigation-results.md). The earlier
unconstrained, single-policy ablations remain in [navigation results](docs/navigation-results.md).

## Retrain with spaced landmarks

New environments default to a minimum pairwise landmark **center distance of
0.4 units**, measured with Euclidean distance in any direction. Complete invalid
layouts are resampled in the original [-1,1] square; landmarks then remain fixed.
Nearby-only sensing stays at radius 0.6. Historical checkpoints without a spacing
setting are evaluated with their original unconstrained layouts.

```powershell
uv sync --extra dev --extra train
uv run --extra train marl-study --output results/spaced-navigation-new
```

This trains three independent GRU MAPPO policies from scratch: full, none, and
fixed communication every five steps. Each receives the same 8,320,000 initial
steps and 640,000 refinement steps, with early stopping disabled. Packet loss
is zero. Each policy is evaluated under its **own** schedule on the same 200
held-out scenarios. Training runs concurrently in three local Python processes,
one PyTorch thread each. Logs and checkpoints are stored under `full/`, `none/`
and `fixed_5/` in the output directory. This differs from the earlier experiment,
which ablated communication on one full-trained policy.

The study produces `report.md`, `summary.json`, per-episode records, and
`comparison.html`, which shows the three separate policies on the first test
scenario. If available, `all-success.html` uses the first scenario in which all
three policies succeeded. Minimum distances and zero-loss packet accounting are
checked during evaluation, and replay metrics are checked against episode logs.

To train just one condition, use `marl-train --schedule full`, `--schedule none`,
or `--schedule fixed --period 5`, with `--min-landmark-distance 0.4`. For matching
checkpoint evaluation, add `--trained-schedule-only` to `marl-evaluate`.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From this directory:

```powershell
uv sync --extra dev
uv run pytest
uv run marl-smoke --agents 3 5 8 --loss 0 0.1 0.3 --episodes 3 --output results/first-smoke.jsonl
```

If Python is unavailable, `uv python install 3.12` installs a runtime.
The runner creates JSONL episode records and a `.metadata.json` sidecar with
configuration and dependency versions. Existing outputs are protected from
overwriting; choose a new output name for each run. `uv.lock` records dependencies.
The default smoke horizon is 25; use `--horizon 100` for the proposed pilot.

## Train and evaluate MAPPO

Install the optional training dependencies first:

```powershell
uv sync --extra dev --extra train
uv run --extra train marl-train --output results/my-training --steps 2000000
```

Training uses three agents, full communication, no loss, radius 0.6 and a
100-step horizon by default. Landmark separation now defaults to 0.4; use
`--min-landmark-distance 0` only to reproduce the old unconstrained experiment.
It runs batched MPE2-equivalent physics on CPU, saving
`best.pt`, `last.pt`, `learning.jsonl` and `config.json`. The shared actor receives
only local observations; the centralized critic receives physical world state
and all agents' pooled inboxes. This is a MAPPO baseline based on
[Yu et al.](https://arxiv.org/abs/2103.01955), not a reproduction of their full code.
Checkpoint selection uses separate validation worlds, with a default target of
80% final-step success. A completed training budget does not imply that target
was reached; inspect the logged validation scores.

The default actor is feedforward. Add `--recurrent --hidden 64` for a GRU actor
that remembers previous local observations. Recurrent PPO trains contiguous
sequences (`--sequence-length 25`); memory resets between episodes and evaluation
conditions. Neither architecture receives hidden landmark coordinates.

Optional training aids are explicitly saved in the configuration:
`--curriculum-steps 1000000` starts some worlds near landmarks, annealing back to
original random starts; `--coverage-bonus 1 --success-bonus 5` rewards coverage;
`--potential-scale 5` adds potential-based shaping from optimal assignment distance.
`--discovery-bonus 5` rewards a previously unseen landmark entering team sensor
range for the first time in an episode.
The default curriculum mixes near-landmark and random starts; `--curriculum-mode
blend` instead interpolates positions from near-landmark starts to random starts.
Validation and evaluation always use original random starts and report base task
rewards without these training additions. Use the exact saved configuration when
reproducing an exploratory run.

Evaluate the **same checkpoint** under full, every-2/5/10-step, and no communication:

```powershell
uv run --extra train marl-evaluate results/my-training/best.pt --episodes 100 --output results/my-evaluation
uv run --extra train marl-visualize --checkpoint results/my-training/best.pt --loss 0 --seed 200000 --output results/trained-demo.html
```

Evaluation uses the original PettingZoo environment and held-out seeds starting
at 200000. It saves per-episode records and a summary with success intervals,
coverage, rewards, collisions and traffic; random navigation is included as a
reference. Evaluation actions are deterministic. No/limited communication results
are ablations of the full-trained policy, not separately trained baselines.

To continue a run with a larger total budget, write a new output directory:

```powershell
uv run --extra train marl-train --resume results/my-training/last.pt --steps 10000000 --output results/my-training-continued
```

Continuation retains model and optimizer weights and step count but starts new
simulator/RNG streams. `--steps` is the cumulative budget, not additional steps.
Resumed runs preserve the source checkpoint as `parent.pt` and record its hash.
All output directories are protected against accidental reuse. Training steps
count summed world steps; agent transitions are three times that count.
Create an empty `STOP` file inside a running training directory to save `last.pt`
and exit at the next PPO update boundary.

## What is here

To test the three existing spaced-landmark checkpoints with 25% packet loss:

```powershell
uv run --extra train python -m limited_comm.loss_study
```

This evaluates frozen policies on the same held-out seeds and preserves the
0.4-unit radial landmark separation. It writes a report and comparison replays
to `results/spaced-navigation-loss25/`. Choose a new `--output` for repeat runs.
See [packet-loss results](docs/spaced-navigation-loss25-results.md).

| File | Purpose |
| --- | --- |
| `docs/research-plan.md` | Hypotheses, experiment design, definitions, next milestones |
| `src/limited_comm/env.py` | Local observations and movement/transmit actions over MPE2 physics |
| `src/limited_comm/channel.py` | Packet loss, message cache, expiration, and traffic accounting |
| `src/limited_comm/experiment.py` | No/full/fixed schedule infrastructure smoke runs |
| `src/limited_comm/train.py` | MAPPO rollout collection, PPO updates, validation and checkpoints |
| `src/limited_comm/evaluate.py` | Original-environment communication ablations and metric reports |
| `src/limited_comm/loss_study.py` | Frozen-checkpoint packet-loss evaluation paired with the spaced-landmark baseline |
| `src/limited_comm/vector_env.py` | Batched training simulation, checked against the original environment |
| `tests/` | Channel semantics, observation isolation, API and metric checks |

## Watch the simulation

Generate a self-contained HTML replay, then open it in your browser:

```powershell
uv run python -m limited_comm.visualize
```

Open `results/demo.html`. It shows the three schedules side by side, with
play/pause, single-step, a timeline slider, speed controls, delivered/dropped
packet arrows, and coverage, collision and cumulative traffic counters.
Blue/colored circles are agents; diamonds are landmarks, turning green when
covered. An optional overlay shows the sensing radius.

Customize with `--agents 5 --loss 0.5 --horizon 200 --period 10 --output results/demo-2.html`.
After `uv sync`, `uv run marl-visualize` is an equivalent entry point.
No server or additional visualization dependencies are required. Navigation is
random and identical across schedules; this visualizes the smoke setup, not RL
training. The HTML replay embeds trajectories rather than running live physics.

## Design choices

Stock [Simple Spread](https://mpe2.farama.org/environments/simple_spread/) provides
navigation and collision physics. Its agents are
[silent in the implementation](https://github.com/Farama-Foundation/MPE2/blob/main/mpe2/simple_spread/simple_spread.py),
so this adapter explicitly adds a communication channel. The MPE environments
[moved out of PettingZoo into MPE2](https://pettingzoo.farama.org/main/environments/mpe/);
the adapter retains the PettingZoo Parallel API.

Each actor sees its own position/velocity, up to three nearby agents and landmarks
within radius 0.6, validity masks, and a pooled inbox. Observation length stays 28
across team sizes. The adapter deliberately rebuilds actor observations from the
world instead of returning the base environment's global observations.

Actions are `[movement, transmit]`: five movement choices and a binary gate.
Messages contain four float32 values: sender position and velocity. Messages sent
from state t arrive in observations at t+1. Delivered messages remain cached for
five steps; their age and availability are observed. Initial inboxes are empty.
Loss is independent for each directed sender/receiver link; no acknowledgments,
congestion, bandwidth cap, or variable latency are modeled.

Each sending agent sends one packet to every other agent (all-to-all **unicast**).
Full communication therefore attempts N(N-1) packets per step, with 16 payload
bytes each. Failed packets still incur cost. These are simulated payload bytes,
excluding headers, and bytes per simulator step rather than physical bits/second.
Inbox pooling bounds policy input size; it does not reduce network fanout or
simulation complexity.

All actors receive the mean base task reward minus `comm_cost * attempted_links / N`.
The centralized `state()` method exposes privileged physical state; the training
critic also receives the pooled inboxes. These critic inputs grow with N. A
critic suitable for transfer across team sizes remains to be implemented.

The navigation policies in smoke runs ignore messages and use paired random
movement. Task returns should match across schedules while traffic and penalized
returns differ. That is an accounting check, not an experiment outcome.
