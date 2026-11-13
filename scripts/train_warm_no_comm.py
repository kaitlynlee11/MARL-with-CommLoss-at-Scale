"""Budget-matched warm-start no-communication controls for adaptive seeds 42-56."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import traceback

import numpy as np

from run_adaptive_seeds import atomic_text, read_json, seed_stats, write_json

REPO = Path(__file__).resolve().parents[1]
ADAPTIVE = REPO / "results/adaptive-zero-loss-15-seeds-20261110"
BASELINE = REPO / "results/baseline-agent-spacing04-15-seeds-20261109"
OUTPUT = REPO / "results/warm-no-communication-15-seeds-20261112"
METRICS = ("success_rate", "final_coverage", "task_return", "collision_step_rate", "collision_episode_fraction",
           "attempted_links", "delivered_links")
GROUPS = ("initial_none", "warm_none", "adaptive")
MODEL_LOCK = threading.Lock()


def patch_trainer(source):
    """Guarded, documented edits to the exact historical training implementation."""
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Frozen trainer differs at patch anchor: {old[:90]}")
        source = source.replace(old, new)
    replace('if args.initialize_actor and (not adaptive or args.resume):',
            'if args.initialize_actor and ((not adaptive and args.schedule != "none") or args.resume):')
    replace('parser.error("initialize-actor requires a fresh adaptive run")',
            'parser.error("initialize-actor requires a fresh adaptive or no-communication control run")')
    # Construct the same original ten-action model before loading the initializer.
    # This preserves the exact adaptive fresh-critic initialization and RNG draws.
    replace('model = MAPPO(hidden=args.hidden, recurrent=args.recurrent, adaptive=adaptive)',
            'model = MAPPO(hidden=args.hidden, recurrent=args.recurrent, adaptive=adaptive or (args.schedule == "none" and args.initialize_actor is not None))')
    old = '''        weights = parent.actor.state_dict()
        # Ten categorical actions: movement + 5 * transmit. Preserve movement marginals,
        # with an initial stochastic transmit probability of 0.1 for every movement.
        weights["head.weight"] = weights["head.weight"].repeat(2, 1)
        bias = weights["head.bias"]
        weights["head.bias"] = torch.cat([bias + np.log(0.9), bias + np.log(0.1)])
        model.actor.load_state_dict(weights)'''
    new = '''        if not adaptive:
            # The same encoder/GRU and five-action movement marginal, with the
            # fresh critic from the same ten-action construction as adaptive.
            model.actor = parent.actor
            model.actor.train()
            model.action_dim = 5
            model.spec["adaptive"] = False
        else:
            weights = parent.actor.state_dict()
            weights["head.weight"] = weights["head.weight"].repeat(2, 1)
            bias = weights["head.bias"]
            weights["head.bias"] = torch.cat([bias + np.log(0.9), bias + np.log(0.1)])
            model.actor.load_state_dict(weights)'''
    replace(old, new)
    replace('if adaptive or args.terminate_on_collision:',
            'if adaptive or args.terminate_on_collision or (args.schedule == "none" and args.initialize_actor):')
    replace('if not adaptive and not args.terminate_on_collision:',
            'if not adaptive and not args.terminate_on_collision and not (args.schedule == "none" and args.initialize_actor):')
    replace('            if update == 1 or update % args.eval_every == 0 or total >= args.steps:',
            '''            if args.schedule == "none" and args.initialize_actor:
                record["training_attempted_links"] = int(rollout_attempts)
                assert record["training_attempted_links"] == 0
            if update == 1 or update % args.eval_every == 0 or total >= args.steps:''')
    replace('if adaptive else "legacy ungated distance baseline"',
            'if adaptive or (args.schedule == "none" and args.initialize_actor) else "legacy ungated distance baseline"')
    replace('    args.output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")',
            '''    config["matched_no_communication_control"] = bool(args.schedule == "none" and args.initialize_actor)
    if config["matched_no_communication_control"]:
        config["stop_rule"] = "fixed per-seed adaptive-matched optimization budget; final last.pt is primary"
        config["initializer_semantics"] = "same full actor movement marginal and adaptive-matched fresh critic; fresh optimizer"
    args.output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "model_spec": model.spec, "config": config,
                "optimizer": optimizer.state_dict(), "steps": 0}, args.output / "initial.pt")''')
    return source


def budget_plan(seed):
    directory = ADAPTIVE / f"seed-{seed}"
    config = read_json(directory / "training/config.json")
    status = read_json(directory / "training/status.json")
    result = read_json(directory / "result.json")
    assert status["state"] == "target_reached" and status["steps"] == result["adaptive_training_steps"]
    assert config["seed"] == seed and config["loss"] == 0
    assert config["min_agent_distance"] == config["min_landmark_distance"] == .4
    assert not config["terminate_on_collision"]
    assert config["curriculum_steps"] == config["potential_scale"] == config["discovery_bonus"] == 0
    rollout_steps = config["envs"] * config["rollout"]
    steps = status["steps"]
    assert steps % rollout_steps == 0
    sequences = rollout_steps // config["sequence_length"]
    sequence_batch = max(1, config["minibatch"] // config["sequence_length"])
    optimizer_steps_per_update = config["epochs"] * math.ceil(sequences / sequence_batch)
    initializer = directory / "training/initializer.pt"
    assert hashlib.sha256(initializer.read_bytes()).hexdigest() == config["initializer_sha256"]
    references = {}
    for name, reference_directory in (("initial_none", BASELINE / f"seed-{seed}/none"), ("adaptive", directory)):
        reference_checkpoint = (reference_directory / "best.pt" if name == "initial_none" else
                                reference_directory / read_json(reference_directory / "selection.json")["checkpoint"])
        expected_hash = (read_json(reference_directory / "selection.json")["sha256"] if name == "initial_none" else
                         result["checkpoint_sha256"])
        assert hashlib.sha256(reference_checkpoint.read_bytes()).hexdigest() == expected_hash
        path = reference_directory / "evaluation/episodes.jsonl"
        references[name] = {"episodes_path": str(path), "episodes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "checkpoint_path": str(reference_checkpoint), "checkpoint_sha256": expected_hash}
    return {"training_seed": seed, "adaptive_config": config, "additional_world_steps": steps,
            "ppo_updates": steps // rollout_steps,
            "optimizer_steps": steps // rollout_steps * optimizer_steps_per_update,
            "initializer_path": str(initializer), "initializer_sha256": config["initializer_sha256"],
            "shared_initializer_training_work": 8960000, "total_training_work": 8960000 + steps,
            "adaptive_checkpoint": str(directory / read_json(directory / "selection.json")["checkpoint"]),
            "adaptive_checkpoint_sha256": result["checkpoint_sha256"], "references": references}


def command(directory, plan, smoke=False):
    c = plan["adaptive_config"]
    steps = c["envs"] * c["rollout"] * 2 if smoke else plan["additional_world_steps"]
    args = ["-m", "limited_comm.train", "--output", str(directory / "training"),
            "--schedule", "none", "--initialize-actor", plan["initializer_path"],
            "--seed", str(plan["training_seed"]), "--steps", str(steps), "--no-early-stop", "--recurrent"]
    keys = ("envs", "rollout", "horizon", "hidden", "sequence_length", "epochs", "minibatch", "lr", "entropy",
            "gradient_clipping", "threads", "eval_every", "validation_episodes", "target_success", "radius",
            "period", "min_landmark_distance", "min_agent_distance", "loss", "coverage_bonus", "success_bonus",
            "collision_penalty", "packet_penalty", "time_penalty", "distance_penalty", "packet_ratio_target",
            "training_packet_budget", "packet_dual_lr")
    for key in keys:
        value = 2 if smoke and key == "validation_episodes" else c[key]
        args.extend(["--" + key.replace("_", "-"), str(value)])
    return args, steps


def check_initialization(directory, plan):
    with MODEL_LOCK:
        return _check_initialization(directory, plan)


def _check_initialization(directory, plan):
    import torch
    from limited_comm.mappo import MAPPO, load_policy
    # Orthogonal initialization uses QR; matching CPU thread count is required
    # for bitwise reconstruction, even with the same random seed.
    torch.set_num_threads(plan["adaptive_config"]["threads"])
    initial, checkpoint = load_policy(directory / "training/initial.pt")
    parent, _ = load_policy(Path(plan["initializer_path"]))
    assert initial.action_dim == 5 and not initial.spec["adaptive"]
    assert not checkpoint["optimizer"]["state"]
    for key, value in initial.actor.state_dict().items():
        assert torch.equal(value, parent.actor.state_dict()[key])
    # Reconstruct the original fresh adaptive critic with the identical seed and
    # construction order. Its actor is subsequently overwritten by warm-start.
    torch.manual_seed(plan["training_seed"])
    reference = MAPPO(hidden=plan["adaptive_config"]["hidden"], recurrent=True, adaptive=True)
    for key, value in initial.critic.state_dict().items():
        assert torch.equal(value, reference.critic.state_dict()[key]), key
    for key in ("value_mean", "value_std", "value_updates"):
        assert torch.equal(getattr(initial, key), getattr(reference, key))
    weights = parent.actor.state_dict()
    weights["head.weight"] = weights["head.weight"].repeat(2, 1)
    bias = weights["head.bias"]
    weights["head.bias"] = torch.cat([bias + np.log(.9), bias + np.log(.1)])
    reference.actor.load_state_dict(weights)
    observations = torch.from_numpy(np.random.default_rng(42).normal(size=(4, 3, 28)).astype(np.float32))
    with torch.no_grad():
        original_probabilities = reference.distribution(observations).probs
        control_probabilities = initial.distribution(observations).probs
        assert torch.allclose(control_probabilities, original_probabilities[..., :5] + original_probabilities[..., 5:], atol=1e-7)
    assert hashlib.sha256((directory / "training/initializer.pt").read_bytes()).hexdigest() == plan["initializer_sha256"]
    return {"actor_exactly_matches_full_initializer": True, "initial_movement_marginal_matches_adaptive": True,
            "fresh_critic_exactly_matches_adaptive_initialization": True,
            "initialization_cpu_threads": plan["adaptive_config"]["threads"],
            "optimizer_started_empty": True, "initializer_sha256": plan["initializer_sha256"]}


def summarize_rows(rows):
    from limited_comm.evaluate import summarize
    return {**summarize(rows), "collision_episode_fraction": float(np.mean([r["any_collision"] for r in rows]))}


def train_seed(root, seed, smoke, jobs):
    import torch
    from limited_comm.mappo import load_policy
    manifest = read_json(root / "manifest.json")
    plan = manifest["plans"][str(seed)]
    directory = root / f"seed-{seed}"
    directory.mkdir()
    jobs[str(seed)] = "training"
    training_command, expected_steps = command(directory, plan, smoke)
    environment = {**os.environ, "PYTHONPATH": str(root / "source"), "OMP_NUM_THREADS": "1"}
    def child(args, log):
        with log.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(list(map(str, args))) + "\n")
            stream.flush()
            subprocess.run([sys.executable, "-u", *map(str, args)], cwd=REPO, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    child(training_command, directory / "training.console.log")
    status = read_json(directory / "training/status.json")
    assert status["state"] == "budget_exhausted" and status["steps"] == expected_steps
    initialization = check_initialization(directory, plan)
    with MODEL_LOCK:
        model, final = load_policy(directory / "training/last.pt")
    assert final["steps"] == expected_steps
    config = final["config"]
    for key in ("envs", "rollout", "horizon", "hidden", "sequence_length", "epochs", "minibatch", "lr", "entropy",
                "gradient_clipping", "radius", "loss", "min_agent_distance", "min_landmark_distance",
                "coverage_bonus", "success_bonus", "collision_penalty", "packet_penalty", "time_penalty", "distance_penalty"):
        assert config[key] == plan["adaptive_config"][key], key
    assert config["no_early_stop"] and config["matched_no_communication_control"] and not model.spec["adaptive"]
    logs = [json.loads(line) for line in (directory / "training/learning.jsonl").read_text().splitlines()]
    expected_updates = expected_steps // (config["envs"] * config["rollout"])
    optimizer_count = expected_updates * config["epochs"] * math.ceil(
        (config["envs"] * config["rollout"] // config["sequence_length"]) /
        max(1, config["minibatch"] // config["sequence_length"]))
    assert all(int(state["step"].item()) == optimizer_count for state in final["optimizer"]["state"].values())
    assert len(logs) == expected_updates and logs[-1]["update"] == expected_updates
    assert all(np.isfinite(r[k]) for r in logs for k in ("policy_loss", "value_loss", "entropy", "approx_kl"))
    assert all(r["training_attempted_links"] == 0 for r in logs)
    assert all(r["validation"]["attempted_packets"] == 0 for r in logs if "validation" in r)
    jobs[str(seed)] = "evaluating"
    episodes = 2 if smoke else 200
    child(["-m", "limited_comm.evaluate", directory / "training/last.pt", "--trained-schedule-only",
           "--episodes", episodes, "--seed", 400000, "--loss", 0, "--min-agent-distance", .4,
           "--output", directory / "evaluation"], directory / "evaluation.console.log")
    records = [json.loads(line) for line in (directory / "evaluation/episodes.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in records] == list(range(400000, 400000 + episodes))
    assert all(r["attempted_links"] == r["delivered_links"] == 0 and not r["initial_collision"]
               and r["active_steps"] == 100 and r["task_reward_version"] == "legacy-distance-v0" for r in records)
    groups = {"warm_none": summarize_rows(records)}
    for name, reference in plan["references"].items():
        path = Path(reference["episodes_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["episodes_sha256"]
        original = [json.loads(line) for line in path.read_text().splitlines()][:episodes]
        assert len(original) == len(records)
        for a, b in zip(original, records):
            assert a["seed"] == b["seed"]
            assert a["actual_initial_agent_separation"] == b["actual_initial_agent_separation"] >= .4
            assert a["actual_min_separation"] == b["actual_min_separation"] >= .4
        groups[name] = summarize_rows(original)
    result = {"training_seed": seed, "groups": groups, "adaptive_matched_world_steps": plan["additional_world_steps"],
              "control_world_steps": expected_steps, "ppo_updates": expected_updates,
              "optimizer_steps": optimizer_count,
              "total_training_work_including_initializer": plan["shared_initializer_training_work"] + expected_steps,
              "initialization_verification": initialization, "checkpoint": "training/last.pt",
              "checkpoint_sha256": hashlib.sha256((directory / "training/last.pt").read_bytes()).hexdigest(),
              "differences": {f"warm_none_minus_{name}": {m: groups["warm_none"][m] - groups[name][m] for m in METRICS}
                              for name in ("initial_none", "adaptive")}}
    write_json(directory / "result.json", result)
    return result


def aggregate(root, rows, seeds, smoke=False, evaluation_episodes=200):
    rows = sorted(rows, key=lambda r: r["training_seed"])
    assert len({r["training_seed"] for r in rows}) == len(rows)
    stats = {name: {m: seed_stats([r["groups"][name][m] for r in rows]) for m in METRICS} for name in GROUPS}
    differences = {name: {m: seed_stats([r["differences"][name][m] for r in rows]) for m in METRICS}
                   for name in ("warm_none_minus_initial_none", "warm_none_minus_adaptive")}
    result = {"complete": len(rows) == len(seeds), "smoke_only": smoke, "training_seeds": seeds,
              "completed_training_seeds": [r["training_seed"] for r in rows], "groups": stats,
              "paired_differences": differences, "per_seed": rows}
    write_json(root / "summary.json", result)
    fields = ["training_seed", "control_world_steps", "ppo_updates", "optimizer_steps", "total_training_work_including_initializer",
              *[f"{g}_{m}" for g in GROUPS for m in METRICS]]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             **{f"{g}_{m}": row["groups"][g][m] for g in GROUPS for m in METRICS}})
    lines = ["# Warm-start no-communication control versus initial none and adaptive" + (" (SMOKE ONLY)" if smoke else ""), "",
        f"Status: {'complete' if result['complete'] else 'in progress'}; {len(rows)}/{len(seeds)} matched policy seeds evaluated.", "",
        f"All comparisons use original training seeds 42-56 and the same {evaluation_episodes} held-out scenarios (400000-{400000 + evaluation_episodes - 1}), with three agents/landmarks, zero loss, 0.4-unit radial agent-start and landmark spacing, 100-step episodes and the original legacy communication/task-return protocol. This study is separate from the exploratory landmark-sharing changes.", "",
        "The new no-communication policy starts from the exact full-communication actor initializer used by adaptive for that seed. Its five-action actor preserves that initializer's movement distribution. To match the fresh adaptive critic exactly, the same seeded ten-action model is constructed first, then only its actor is replaced with the initializer's five-action movement actor. The critic, optimizer and PopArt state are fresh. Saved initial.pt verifies actor equality, fresh-critic equality and an empty optimizer.", "",
        "Each control trains with zero transmit gates throughout, the adaptive navigation reward (coverage/full-coverage bonuses, collision/time/distance terms; packet term is zero), and the identical PPO hyperparameters, batch/rollout/sequence sizes and number of epochs. Each seed's actual adaptive steps and optimizer update count are matched, rather than the unused 100-million-step ceiling. Early stopping is disabled. The primary control policy is final last.pt at the matched budget, matching adaptive's final target.pt update; validation-best artifacts are retained but not used to select the reported control. Held-out results never select checkpoints.", "",
        "Initial none is the separately trained from-scratch 8,960,000-step baseline. Warm none and adaptive additionally share the same initializer-training cost and seed-specific optimization work. Initial none is an original reference, not an equal-compute condition. The movement-only control has five actions, while adaptive has ten joint movement/transmit actions; no dummy transmit actions or extra exploration choices are added to the control.", "",
        "Every entry is mean [95% training-seed bootstrap interval], using 20,000 percentile resamples of equally weighted seeds. Partial summaries compare only matching completed seeds across all three groups. Task return uses baseline legacy-distance-v0 scoring, excludes communication penalties and shaping bonuses, and is better when less negative.", "",
        "| Policy | Seeds | Success | Coverage | Collision step rate | Task return | Packets/episode |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    def cell(s, percent=False):
        def fmt(v):
            return f"{v:.2%}" if percent else f"{v:.4f}"
        ci = s.get("seed_bootstrap_95pct")
        return fmt(s["mean"]) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else " [pending]")
    if rows:
        for name in GROUPS:
            m = stats[name]
            lines.append(f"| {name} | {len(rows)} | {cell(m['success_rate'],True)} | {cell(m['final_coverage'],True)} | {cell(m['collision_step_rate'],True)} | {cell(m['task_return'])} | {m['attempted_links']['mean']:.4f} |")
        lines += ["", "## Paired differences (warm none minus reference)", "",
                  "| Reference | Success change (pp) | Collision step rate change (pp) | Return change |",
                  "| --- | --- | --- | --- |"]
        for name, metrics in differences.items():
            def delta(s, percent=False):
                ci = s.get("seed_bootstrap_95pct")
                scale = 100 if percent else 1
                return f"{scale*s['mean']:+.4f}" + (f" [{scale*ci[0]:+.4f}, {scale*ci[1]:+.4f}]" if ci else " [pending]")
            lines.append(f"| {name} | {delta(metrics['success_rate'],True)} | {delta(metrics['collision_step_rate'],True)} | {delta(metrics['task_return'])} |")
    lines += ["", "## Seed-specific optimization budgets", "",
        "Full plan and initializer/adaptive-checkpoint hashes: manifest.json. Each seed inherits 8,960,000 world steps of initializer training work, then matches that seed's completed adaptive optimization steps. Actor/critic/optimizer initialization checks and actual PPO update counts are saved in seed-N/result.json.", "",
        "Individual outcomes: per-seed.csv, summary.json and seed-N/evaluation/episodes.jsonl. Training curves: seed-N/training/learning.jsonl. The auto-refreshing progress.html distinguishes budget progress from policy performance. Historical initial-none and adaptive results are reused, never overwritten or retrained.", ""]
    atomic_text(root / "report.md", "\n".join(lines))
    if not smoke:
        atomic_text(REPO / "docs/warm-no-communication-results.md", "\n".join(lines))


def progress(root, manifest, jobs, smoke=False):
    from serve_progress import tail_records
    runs = []
    for seed in manifest["training_seeds"]:
        plan = manifest["plans"][str(seed)]
        tail = tail_records(root / f"seed-{seed}/training/learning.jsonl")
        last = (tail or [{}])[-1]
        val = next((r["validation"] for r in reversed(tail) if "validation" in r), {})
        goal = plan["adaptive_config"]["envs"] * plan["adaptive_config"]["rollout"] * 2 if smoke else plan["additional_world_steps"]
        runs.append({"seed": seed, "state": jobs[str(seed)], "steps": last.get("steps", 0),
                     "budget": goal, "validation": val, "result": read_json(root / f"seed-{seed}/result.json")})
    state = "complete" if all(r["state"] == "complete" for r in runs) else "failed" if any(r["state"] == "failed" for r in runs) and not any(r["state"] in ("training", "queued", "evaluating") for r in runs) else "running"
    write_json(root / "status.json", {"state": state, "updated_unix": time.time(), "jobs": dict(jobs), "runs": runs,
               "completed_seeds": sum(r["state"] == "complete" for r in runs), "expected_seeds": len(runs)})
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>Matched warm-start no-communication training</title><style>body{max-width:1100px;margin:40px auto;padding:0 22px;background:#f5f6f9;color:#203047;font:14px system-ui}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}progress{width:100%;accent-color:#5366ce}a{color:#5366ce}</style><h1>Warm-start no-communication controls</h1><p>Same initializer and per-seed adaptive optimization budget · seeds 42-56 · zero loss · 0.4-unit spacing</p>'''
    page += f"<p>Updated {time.strftime('%H:%M:%S UTC',time.gmtime())} · {state}</p><p><a href='report.md'>Comparison report</a> · <a href='per-seed.csv'>Per-seed results</a></p><table><tr><th>Seed</th><th>State</th><th>Matched optimization budget</th><th>Validation success</th><th>Validation collision rate</th><th>Held-out success</th></tr>"
    for r in runs:
        v, result = r["validation"], r["result"]
        success = f"{v['success']:.1%}" if v else "pending"
        collision = f"{v['collision_step_rate']:.2%}" if v else "pending"
        heldout = f"{result['groups']['warm_none']['success_rate']:.1%}" if result else "pending"
        page += f"<tr><td>{r['seed']}</td><td>{r['state']}</td><td>{r['steps']:,} / {r['budget']:,}<progress value='{r['steps']}' max='{r['budget']}'></progress></td><td>{success}</td><td>{collision}</td><td>{heldout}</td></tr>"
    page += "</table><p>Training cannot stop early on success. Primary evaluation uses the final checkpoint at the matched optimizer budget. Initial-none and adaptive policies are saved references.</p></html>"
    try:
        atomic_text(root / "progress.html", page)
    except OSError as error:
        print(f"Display update deferred: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error("workers must be positive")
    seeds = [42] if cli.smoke else list(range(42, 57))
    plans = {str(seed): budget_plan(seed) for seed in seeds}
    root = cli.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_manifest = read_json(ADAPTIVE / "manifest.json")
    shutil.copytree(ADAPTIVE / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    for name, digest in source_manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    trainer = root / "source/limited_comm/train.py"
    (root / "original-train.py").write_bytes(trainer.read_bytes())
    trainer.write_text(patch_trainer(trainer.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copyfile(Path(__file__), root / "runner.py")
    manifest = {"training_seeds": seeds, "plans": plans, "smoke_only": cli.smoke, "workers": cli.workers,
                "source_adaptive": str(ADAPTIVE), "source_initial_none": str(BASELINE),
                "communication_protocol": "position-velocity-all-to-all-v0", "loss": 0.,
                "min_agent_distance": .4, "min_landmark_distance": .4,
                "source_sha256": {p.relative_to(root / "source").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "source").rglob("*.py")},
                "original_source_sha256": source_manifest["source_sha256"],
                "changes": "Only train.py modified: permit movement-only warm start, match fresh adaptive critic initialization, use adaptive navigation reward under zero traffic, save initial state. Simulator and all optimization update code unchanged.",
                "primary_checkpoint": "training/last.pt at matched per-seed actual adaptive steps",
                "total_additional_world_steps": sum(p["additional_world_steps"] for p in plans.values())}
    write_json(root / "manifest.json", manifest)
    sys.path.insert(0, str(root / "source"))
    import torch
    torch.set_num_threads(1)
    from limited_comm.mappo import load_policy
    for plan in plans.values():
        checkpoint_path = Path(plan["adaptive_checkpoint"])
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == plan["adaptive_checkpoint_sha256"]
        _, checkpoint = load_policy(checkpoint_path)
        assert checkpoint["steps"] == plan["additional_world_steps"]
        assert all(int(s["step"].item()) == plan["optimizer_steps"] for s in checkpoint["optimizer"]["state"].values())
    manifest["adaptive_final_optimizer_counters_verified"] = True
    write_json(root / "manifest.json", manifest)
    rows, jobs = [], {str(seed): "queued" for seed in seeds}
    aggregate(root, rows, seeds, cli.smoke)
    progress(root, manifest, jobs, cli.smoke)
    with ThreadPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(train_seed, root, seed, cli.smoke, jobs): seed for seed in seeds}
        pending = set(futures)
        while pending:
            for future in list(pending):
                if future.done():
                    pending.remove(future)
                    seed = futures[future]
                    try:
                        rows.append(future.result())
                        jobs[str(seed)] = "complete"
                    except Exception:
                        jobs[str(seed)] = "failed"
                        write_json(root / f"seed-{seed}/failure.json", {"traceback": traceback.format_exc()})
                    aggregate(root, rows, seeds, cli.smoke)
            progress(root, manifest, jobs, cli.smoke)
            if pending:
                time.sleep(5)
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for r in rows:
            for line in (root / f"seed-{r['training_seed']}/evaluation/episodes.jsonl").read_text().splitlines():
                stream.write(json.dumps({"training_seed": r["training_seed"], **json.loads(line)}) + "\n")
    if len(rows) != len(seeds):
        raise SystemExit("Some matched controls failed; inspect failure.json")


if __name__ == "__main__":
    main()
