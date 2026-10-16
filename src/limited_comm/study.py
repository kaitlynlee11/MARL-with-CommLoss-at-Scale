"""Train three independent spaced-landmark baselines and compare their own schedules.

Workers are Python training processes, each with its own RNG and model. All three
receive identical budgets and hyperparameters; no checkpoint is shared between
communication conditions. Logs live in each condition's directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .mappo import load_policy
from .visualize import HTML, collect_replay


def child(command, logfile):
    with logfile.open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", *command], stdout=stream, stderr=subprocess.STDOUT, check=True)


def train_condition(label, schedule, root, args):
    directory = root / label
    directory.mkdir()
    common = ["--schedule", schedule, "--period", "5", "--min-landmark-distance", str(args.min_landmark_distance),
              "--seed", str(args.training_seed), "--threads", "1", "--recurrent", "--hidden", "64",
              "--eval-every", "50", "--no-early-stop", "--coverage-bonus", "1", "--potential-scale", "5"]
    initial = directory / "initial"
    print(json.dumps({"condition": label, "phase": "initial", "status": "started"}), flush=True)
    child(["-m", "limited_comm.train", "--output", str(initial), "--steps", str(args.initial_steps),
           "--curriculum-steps", str(args.curriculum_steps), "--entropy", "0.002", "--success-bonus", "5",
           "--gradient-clipping", "joint", *common], directory / "initial.console.log")
    _, parent = load_policy(initial / "last.pt")
    refine = directory / "refine"
    print(json.dumps({"condition": label, "phase": "refine", "status": "started"}), flush=True)
    child(["-m", "limited_comm.train", "--resume", str(initial / "last.pt"), "--output", str(refine),
           "--steps", str(parent["steps"] + args.refine_steps), "--entropy", "0.001", "--success-bonus", "10",
           "--gradient-clipping", "separate", "--lr", "0.0003", "--sequence-length", "100",
           "--minibatch", "6400", "--epochs", "3", *common], directory / "refine.console.log")
    candidates = []
    for path in (initial / "best.pt", refine / "best.pt"):
        _, checkpoint = load_policy(path)
        candidates.append((checkpoint["validation"]["success"], checkpoint["validation"]["coverage"], path, checkpoint))
    _, _, selected_path, selected = max(candidates, key=lambda item: item[:2])
    destination = directory / "best.pt"
    shutil.copyfile(selected_path, destination)
    selection = {"training_schedule": schedule, "period": 5, "source": str(selected_path),
                 "selected_steps": selected["steps"], "validation": selected["validation"],
                 "initial_work_steps": parent["steps"],
                 "total_work_steps": parent["steps"] + args.refine_steps,
                 "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (directory / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"condition": label, "phase": "evaluation", "validation": selected["validation"]}), flush=True)
    child(["-m", "limited_comm.evaluate", str(destination), "--trained-schedule-only", "--episodes", str(args.episodes),
           "--seed", str(args.test_seed), "--loss", "0", "--output", str(directory / "evaluation")],
          directory / "evaluation.console.log")
    return label, selection


def finalize(root, args, selections, seconds):
    summaries, records = {}, []
    models = {}
    for label, schedule in [("full", "full"), ("none", "none"), ("fixed_5", "fixed")]:
        directory = root / label
        summary = json.loads((directory / "evaluation/summary.json").read_text(encoding="utf-8"))
        summaries[label] = summary["conditions"][label]
        rows = [json.loads(line) for line in (directory / "evaluation/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == args.episodes
        assert all(r["actual_min_separation"] >= args.min_landmark_distance for r in rows)
        assert all(r["loss"] == 0 and r["attempted_links"] == r["delivered_links"] for r in rows)
        records.extend(rows)
        models[schedule] = load_policy(directory / "best.pt")[0]
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    result = {"conditions": summaries, "selections": selections, "min_landmark_distance": args.min_landmark_distance,
              "loss": 0.0, "radius": 0.6, "horizon": 100, "episodes_per_condition": args.episodes,
              "training_seed": args.training_seed, "test_seed_start": args.test_seed,
              "scope": "three independently trained policies, each evaluated under its training schedule",
              "training_work_steps_per_policy": args.initial_steps + args.refine_steps, "seconds": seconds}
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    success_sets = [{r["seed"] for r in records if r["schedule"] == s and r["final_success"]}
                    for s in ("full", "none", "fixed")]
    common_success = set.intersection(*success_sets)
    replay_seeds = {"comparison.html": args.test_seed}
    if common_success:
        replay_seeds["all-success.html"] = min(common_success)
    for filename, seed in replay_seeds.items():
        replay = collect_replay(3, 0.0, 100, 5, seed, min_landmark_distance=args.min_landmark_distance, models=models)
        for run in replay["runs"]:
            record = next(r for r in records if r["seed"] == seed and r["schedule"] == run["schedule"])
            assert run["frames"][-1]["success"] == record["final_success"]
            assert abs(run["frames"][-1]["task_return"] - record["task_return"]) < 1e-6
        data = json.dumps(replay, allow_nan=False).replace("<", "\\u003c")
        (root / filename).write_text(HTML.replace("__REPLAY_DATA__", data), encoding="utf-8")
    lines = ["# Spaced-landmark MAPPO baselines", "",
             "Three policies were trained independently from scratch, with identical two-phase budgets and hyperparameters.",
             "Each policy was evaluated under its own training schedule on the same held-out scenario seeds.",
             f"Pairwise landmark center distance >= {args.min_landmark_distance}; loss=0; sensing radius=0.6; horizon=100.",
             f"Training work: {args.initial_steps + args.refine_steps:,} world steps per policy. Training seed: {args.training_seed}.",
             f"Held-out episodes per policy: {args.episodes}; seeds {args.test_seed} through {args.test_seed + args.episodes - 1}.", "",
             "| Training and evaluation schedule | Final success (95% episode interval) | Coverage | Task return | Packets/episode | Collision step rate |",
             "| --- | --- | --- | --- | --- | --- |"]
    for label, summary in summaries.items():
        lo, hi = summary["success_95pct_wilson"]
        lines.append(f"| {label} | {summary['success_rate']:.1%} ({lo:.1%} to {hi:.1%}) | "
                     f"{summary['final_coverage']:.1%} | {summary['task_return']:.2f} | "
                     f"{summary['attempted_links']:.0f} | {summary['collision_step_rate']:.1%} |")
    lines.extend(["", "Success requires simultaneous coverage of all three landmarks at the final step (center distance <= 0.1).",
                  "All training, curriculum, validation, evaluation and replay layouts obey the same spacing rule; landmarks never move.",
                  "The smallest observed held-out landmark distance was " + f"{min(r['actual_min_separation'] for r in records):.6f}.",
                  "No hidden landmark coordinates are supplied to actors. Inputs remain the original 28 values; GRU memory retains past observations.",
                  "Messages contain position/velocity only (16 payload bytes). Cost is reported separately from the base task return.",
                  "Training rewards add one per covered landmark, an all-covered bonus of 5 initially/10 in refinement, and assignment-potential scale 5.",
                  f"A near-landmark initial-state curriculum is annealed to original random agent starts by {args.curriculum_steps:,} steps.",
                  "All schedules use a 64-unit GRU. Phase 1: LR 0.0005, entropy 0.002, 5 epochs, 25-step sequences, joint clipping.",
                  "Phase 2: LR 0.0003, entropy 0.001, 3 epochs, 100-step sequences, separate clipping. Early stopping is disabled for equal budgets.",
                  "Final checkpoints are selected by validation success then coverage, comparing the best checkpoints from both phases.",
                  "This is one training seed per condition, not a multi-seed study. Intervals measure episode uncertainty only.",
                  "These results use a changed layout distribution and independently trained policies; they are not directly comparable to the older test-time ablations.",
                  "", "Local artifacts: each condition's best.pt, selection.json, initial/refine/config.json and learning.jsonl; root summary.json and episodes.jsonl.",
                  "comparison.html uses the first held-out seed; all-success.html, if present, selects the first held-out seed where all three policies succeeded."])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "conditions": summaries, "output": str(root.resolve())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/spaced-navigation"))
    parser.add_argument("--initial-steps", type=int, default=8_320_000)
    parser.add_argument("--refine-steps", type=int, default=640_000)
    parser.add_argument("--curriculum-steps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=400_000)
    parser.add_argument("--min-landmark-distance", type=float, default=0.4)
    args = parser.parse_args()
    if (min(args.initial_steps, args.refine_steps, args.episodes) < 1 or args.initial_steps % 6400
            or args.refine_steps % 6400 or not np.isfinite(args.min_landmark_distance) or args.min_landmark_distance < 0):
        parser.error("positive budgets divisible by 6400, positive episodes and valid spacing are required")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifest.json").write_text(json.dumps({**vars(args), "output": str(root), "schedules": ["full", "none", "fixed_5"]}, indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    started = time.perf_counter()
    selections = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_condition, label, schedule, root, args)
                   for label, schedule in [("full", "full"), ("none", "none"), ("fixed_5", "fixed")]]
        for future in as_completed(futures):
            label, selection = future.result()
            selections[label] = selection
    finalize(root, args, selections, time.perf_counter() - started)


if __name__ == "__main__":
    main()
