# Adaptive communication scheduling for cooperative navigation

Starter project for Kaitlyn Lee's EN.665.645 research proposal. The goal is to test
whether learned communication timing improves the tradeoff between coordination
and communication cost as teams grow and messages are lost.

**Current milestone:** runnable environment, lossy channel, baseline schedules,
metrics, reproducible smoke runner, and tests. MAPPO training and a learned
scheduler are not implemented yet. Smoke results use random movement and cannot
support claims about adaptive scheduling.

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

## What is here

| File | Purpose |
| --- | --- |
| `docs/research-plan.md` | Hypotheses, experiment design, definitions, next milestones |
| `src/limited_comm/env.py` | Local observations and movement/transmit actions over MPE2 physics |
| `src/limited_comm/channel.py` | Packet loss, message cache, expiration, and traffic accounting |
| `src/limited_comm/experiment.py` | No/full/fixed schedule infrastructure smoke runs |
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
The centralized `state()` method exposes privileged physical state for future
critic training only; its dimension grows with N. A critic suitable for transfer
across team sizes remains to be implemented.

The navigation policies in smoke runs ignore messages and use paired random
movement. Task returns should match across schedules while traffic and penalized
returns differ. That is an accounting check, not an experiment outcome.
