"""Evaluate one fixed checkpoint under full, periodic and absent communication."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .env import CommunicationSpread
from .mappo import load_policy
from .layout import minimum_separation
from .schedules import transmit


def wilson(successes, count):
    p, z = successes / count, 1.96
    denominator = 1 + z * z / count
    middle = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(middle - half), float(middle + half)]


def evaluate_episode(model, seed, schedule, period=5, loss=0.0, horizon=100, radius=0.6, random=False,
                     min_landmark_distance=0.4):
    env = CommunicationSpread(horizon=horizon, radius=radius, loss=loss, comm_cost=0.0,
                              min_landmark_distance=min_landmark_distance)
    rng = np.random.default_rng(seed)
    attempts = delivered = collisions = collision_steps = payload_bytes = 0
    task_return = 0.0
    first_success = None
    try:
        obs, _ = env.reset(seed=seed)
        actual_separation = float(minimum_separation(np.asarray([l.state.p_pos for l in env.world.landmarks])))
        if actual_separation < min_landmark_distance:
            raise AssertionError("Evaluation layout violates minimum landmark distance")
        model.reset_memory()
        for t in range(horizon):
            moves = rng.integers(5, size=3) if random else model.moves(np.stack([obs[a] for a in env.possible_agents]))
            gate = transmit(schedule, t, period)
            obs, _, _, _, infos = env.step({a: np.asarray([moves[i], gate]) for i, a in enumerate(env.possible_agents)})
            info = infos[env.possible_agents[0]]
            attempts += info["attempted_links"]
            delivered += info["delivered_links"]
            payload_bytes += info["payload_bytes"]
            collisions += info["collision_pairs"]
            collision_steps += int(info["collision_pairs"] > 0)
            task_return += info["task_reward"]
            if info["success"] and first_success is None:
                first_success = t + 1
        return {"seed": seed, "schedule": schedule, "period": period, "loss": loss,
                "random": random, "final_success": info["success"], "final_coverage": info["coverage"],
                "min_landmark_distance": min_landmark_distance, "actual_min_separation": actual_separation,
                "first_success_step": first_success, "mean_landmark_distance": info["mean_landmark_distance"],
                "task_return": task_return, "attempted_links": attempts, "delivered_links": delivered,
                "payload_bytes": payload_bytes, "collision_pairs": collisions,
                "collision_step_rate": collision_steps / horizon}
    finally:
        env.close()


def summarize(records):
    count = len(records)
    successes = sum(r["final_success"] for r in records)
    return {"episodes": count, "success_rate": successes / count,
            "success_95pct_wilson": wilson(successes, count),
            "ever_success_rate": sum(r["first_success_step"] is not None for r in records) / count,
            **{key: float(np.mean([r[key] for r in records])) for key in
               ["final_coverage", "mean_landmark_distance", "task_return", "attempted_links",
                "delivered_links", "payload_bytes", "collision_step_rate"]}}


def render_report(result):
    summaries = result["conditions"]
    count = next(iter(summaries.values()))["episodes"]
    matching = result.get("evaluation_mode") == "trained_schedule_only"
    training_schedule = result.get("training_config", {}).get("training_schedule", "full")
    lines = ["# MAPPO evaluation" if matching else "# MAPPO communication comparison", "",
             f"Checkpoint: `{result['checkpoint']}` ({result['checkpoint_steps']:,} world steps).",
             f"{count} paired held-out episodes per condition; deterministic actions; loss={result['loss']}.", "",
             "| Communication | Final success (95% episode interval) | Coverage | Task return | Attempted packets | Collision step rate |",
             "| --- | --- | --- | --- | --- | --- |"]
    for label, summary in summaries.items():
        lo, hi = summary["success_95pct_wilson"]
        lines.append(f"| {label} | {summary['success_rate']:.1%} ({lo:.1%}–{hi:.1%}) | "
                     f"{summary['final_coverage']:.1%} | {summary['task_return']:.2f} | "
                     f"{summary['attempted_links']:.0f} | {summary['collision_step_rate']:.1%} |")
    lines.extend(["", "Policy evaluated under its own training schedule." if matching else
                  f"All learned-policy rows use the same {training_schedule}-communication-trained checkpoint. The random_full row uses random movement as a reference.",
                  "Intervals quantify evaluation-episode uncertainty for this checkpoint; they do not measure variation across training seeds.",
                  "This is not an adaptive scheduler study."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--loss", type=float, default=0.0)
    parser.add_argument("--trained-schedule-only", action="store_true", help="Evaluate under this checkpoint's own training schedule")
    parser.add_argument("--output", type=Path, default=Path("results/mappo-evaluation"))
    args = parser.parse_args()
    if args.episodes < 1 or min(args.periods) < 1 or not 0 <= args.loss <= 1:
        parser.error("require positive episodes/periods and loss in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, checkpoint = load_policy(args.checkpoint)
    conditions = [("full", "full", 1, False), *[(f"fixed_{p}", "fixed", p, False) for p in args.periods],
                  ("none", "none", 1, False), ("random_full", "full", 1, True)]
    if args.trained_schedule_only:
        schedule = checkpoint["config"].get("training_schedule", "full")
        period = checkpoint["config"].get("period", 5)
        label = f"fixed_{period}" if schedule == "fixed" else schedule
        conditions = [(label, schedule, period, False)]
    summaries = {}
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as file:
        for label, schedule, period, random in conditions:
            records = []
            for i in range(args.episodes):
                record = evaluate_episode(model, args.seed + i, schedule, period, args.loss,
                                          checkpoint["config"]["horizon"], checkpoint["config"]["radius"], random,
                                          checkpoint["config"].get("min_landmark_distance", 0.0))
                records.append(record)
                file.write(json.dumps({"condition": label, **record}) + "\n")
            summaries[label] = summarize(records)
            print(label, json.dumps(summaries[label]), flush=True)
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_steps": checkpoint["steps"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "seed_start": args.seed, "loss": args.loss, "deterministic_actions": True,
              "training_config": checkpoint["config"], "conditions": summaries,
              "evaluation_mode": "trained_schedule_only" if args.trained_schedule_only else "communication_ablations",
              "scope": "own training schedule" if args.trained_schedule_only else
                       "one trained policy under communication ablations; not separately retrained baselines"}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(render_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
