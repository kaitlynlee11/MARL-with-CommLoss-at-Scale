"""Infrastructure smoke runs. These are NOT trained-policy comparisons."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from .env import CommunicationSpread


def run_episode(n, loss, schedule, period, horizon, seed, comm_cost):
    env = CommunicationSpread(n=n, loss=loss, horizon=horizon, comm_cost=comm_cost)
    rng = np.random.default_rng(seed)
    totals = dict(broadcasts=0, attempted_links=0, delivered_links=0, payload_bytes=0,
                  collision_pairs=0, collision_steps=0, task_return=0.0, penalized_return=0.0)
    first_success = None
    try:
        observations, _ = env.reset(seed=seed)
        for t in range(horizon):
            # Identical random movement across schedules deliberately isolates accounting.
            # No navigation or scheduler learning happens in this script.
            gate = int(schedule == "full" or (schedule == "fixed" and t % period == 0))
            actions = {a: np.asarray([rng.integers(5), gate]) for a in env.agents}
            observations, rewards, _, _, infos = env.step(actions)
            info = infos[env.possible_agents[0]]
            for key in ("broadcasts", "attempted_links", "delivered_links", "payload_bytes", "collision_pairs"):
                totals[key] += info[key]
            totals["collision_steps"] += int(info["collision_pairs"] > 0)
            totals["task_return"] += info["task_reward"]
            totals["penalized_return"] += float(np.mean(list(rewards.values())))
            if info["success"] and first_success is None:
                first_success = t + 1
            if not env.agents:
                break
        steps = env.steps
        return {"n": n, "loss": loss, "schedule": schedule, "seed": seed,
                "steps": steps, **totals, "final_success": info["success"],
                "first_success_step": first_success, "final_coverage": info["coverage"],
                "collision_step_rate": totals["collision_steps"] / steps,
                "collision_pair_rate": totals["collision_pairs"] / (steps * n * (n - 1) / 2),
                "payload_bytes_per_step": totals["payload_bytes"] / steps}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", type=int, default=[3])
    parser.add_argument("--loss", nargs="+", type=float, default=[0.0, 0.3])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm-cost", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("results/smoke.jsonl"))
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.period) < 1:
        parser.error("episodes, horizon and period must be positive")
    if any(n < 2 for n in args.agents) or any(not 0 <= p <= 1 for p in args.loss) or args.comm_cost < 0:
        parser.error("require N >= 2, loss in [0, 1], and nonnegative communication cost")
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error("output already exists; use a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**vars(args), "output": str(args.output), "policy": "random_navigation",
                "purpose": "infrastructure_smoke_test", "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["mpe2", "numpy", "pettingzoo", "gymnasium"]},
                "radius": 0.6, "neighbors": 3, "ttl": 5, "coverage_radius": 0.1,
                "topology": "all_to_all_unicast", "payload_float32_count": 4}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    count = 0
    with args.output.open("x", encoding="utf-8") as file:
        for n in args.agents:
            for loss in args.loss:
                for schedule in ("none", "full", "fixed"):
                    for episode in range(args.episodes):
                        record = run_episode(n, loss, schedule, args.period, args.horizon,
                                             args.seed + episode, args.comm_cost)
                        file.write(json.dumps(record) + "\n")
                        count += 1
    print(f"Wrote {count} smoke episodes to {args.output}; no learned policies evaluated.")


if __name__ == "__main__":
    main()
