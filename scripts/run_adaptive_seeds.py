"""Train and track 15 zero-loss adaptive policies against matched baselines.

Each seed warm-starts its own corrected full-communication navigation actor.
Selection and stopping use validation/confirmation only. Final held-out scores
are never fed back into training, selection or target stopping.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO / "results/baseline-agent-spacing04-15-seeds-20261109"
DEFAULT_OUTPUT = REPO / "results/adaptive-zero-loss-15-seeds-20261110"
METRICS = ("success_rate", "final_coverage", "task_return", "attempted_links", "collision_step_rate")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_text(path, text):
    """Readers on Windows can briefly prevent replacing an open file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False))


def targets_met(success, packets):
    return success > .9 and packets < 120.


def seed_stats(values):
    values = np.asarray(values, dtype=float)
    result = {"mean": float(values.mean()) if len(values) else None,
              "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None}
    if len(values) > 1:
        means = np.random.default_rng(20261110).choice(values, (20000, len(values)), replace=True).mean(axis=1)
        result["seed_bootstrap_95pct"] = np.quantile(means, [.025, .975]).tolist()
    return result


def training_command(seed, directory, initializer, steps, episodes):
    return ["-m", "limited_comm.train", "--output", str(directory / "training"),
            "--schedule", "adaptive", "--initialize-actor", str(initializer),
            "--seed", str(seed), "--steps", str(steps), "--loss", "0",
            "--min-agent-distance", ".4", "--min-landmark-distance", ".4",
            "--recurrent", "--hidden", "64", "--threads", "1", "--lr", ".0001",
            "--epochs", "3", "--entropy", ".002", "--sequence-length", "25",
            "--minibatch", "2048", "--eval-every", "25",
            "--validation-episodes", str(episodes), "--confirmation-episodes", str(episodes),
            "--target-success", ".9", "--packet-ratio-target", ".2",
            "--coverage-bonus", "1", "--success-bonus", "10",
            "--collision-penalty", "5", "--packet-penalty", ".05",
            "--time-penalty", ".01", "--distance-penalty", ".5",
            "--packet-dual-lr", ".02", "--training-packet-budget", ".18"]


def preflight(baseline, seeds):
    """Require measured, completed, identically spaced seed-matched baselines."""
    from limited_comm.mappo import load_policy
    summary = read_json(baseline / "summary.json", {})
    if not summary.get("complete") or summary.get("smoke_only"):
        raise ValueError("Need a completed production baseline study")
    if summary.get("min_agent_distance") != .4 or summary.get("min_landmark_distance") != .4:
        raise ValueError("Baseline must use both radial separation constraints of 0.4")
    initializers = {}
    for seed in seeds:
        directory = baseline / f"seed-{seed}/full"
        path = directory / "best.pt"
        _, checkpoint = load_policy(path)
        config = checkpoint["config"]
        expected = {"seed": seed, "training_schedule": "full", "min_agent_distance": .4,
                    "min_landmark_distance": .4, "loss": 0., "horizon": 100, "radius": .6}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Incompatible initializer for seed {seed}")
        if config.get("terminate_on_collision", False):
            raise ValueError("Adaptive study must match full-horizon baseline success definition")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != read_json(directory / "selection.json", {})["sha256"]:
            raise ValueError(f"Initializer hash mismatch for seed {seed}")
        initializers[seed] = {"path": str(path), "sha256": digest,
                              "baseline_training_work_steps": 8960000,
                              "selected_initializer_steps": checkpoint["steps"]}
    return initializers


def evaluate_and_verify(directory, seed, baseline, initializer, episodes):
    from limited_comm.evaluate import summarize
    summary = read_json(directory / "evaluation/summary.json")
    config = summary["training_config"]
    for key, value in {"seed": seed, "training_schedule": "adaptive", "loss": 0.,
                       "min_agent_distance": .4, "min_landmark_distance": .4,
                       "horizon": 100, "terminate_on_collision": False}.items():
        assert config[key] == value, (key, config.get(key))
    assert config["initializer_sha256"] == initializer["sha256"]
    assert hashlib.sha256((directory / "training/initializer.pt").read_bytes()).hexdigest() == initializer["sha256"]
    assert config["task_reward_version"] == "legacy-distance-v0"
    checkpoint = directory / read_json(directory / "selection.json")["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    records = [json.loads(line) for line in (directory / "evaluation/episodes.jsonl").read_text().splitlines()]
    assert len(records) == episodes
    assert [r["seed"] for r in records] == list(range(400000, 400000 + episodes))
    assert all(r["actual_initial_agent_separation"] >= .4 and r["actual_min_separation"] >= .4
               and not r["initial_collision"] and r["active_steps"] == 100 and not r["terminate_on_collision"]
               and r["loss"] == 0 and r["attempted_links"] == r["delivered_links"]
               and 0 <= r["attempted_links"] <= 600 and 0 <= r["collision_step_rate"] <= 1
               and r["task_reward_version"] == "legacy-distance-v0" for r in records)
    comparisons = {}
    for label in ("full", "none", "fixed_5"):
        baseline_dir = baseline / f"seed-{seed}" / label / "evaluation"
        originals = [json.loads(line) for line in (baseline_dir / "episodes.jsonl").read_text().splitlines()][:episodes]
        assert len(originals) == episodes
        for row, original in zip(records, originals):
            assert row["seed"] == original["seed"]
            assert row["actual_initial_agent_separation"] == original["actual_initial_agent_separation"]
            assert row["actual_min_separation"] == original["actual_min_separation"]
        base = summarize(originals)
        comparisons[label] = {"baseline_metrics": base,
                              "success_to_failure": sum(a["final_success"] and not b["final_success"] for a, b in zip(originals, records)),
                              "failure_to_success": sum(not a["final_success"] and b["final_success"] for a, b in zip(originals, records))}
    metrics = summarize(records)
    assert all(np.isclose(metrics[m], summary["conditions"]["adaptive"][m]) for m in METRICS)
    status = read_json(directory / "training/status.json")
    result = {"training_seed": seed, **metrics,
              "heldout_targets_met": targets_met(metrics["success_rate"], metrics["attempted_links"]),
              "confirmation_targets_met": status["target_met"], "training_stop_reason": status["state"],
              "adaptive_training_steps": status["steps"], "initializer": initializer,
              "max_packets_in_episode": max(r["attempted_links"] for r in records),
              "episodes_at_or_above_120_packets": sum(r["attempted_links"] >= 120 for r in records),
              "comparisons": comparisons, "checkpoint_sha256": summary["checkpoint_sha256"]}
    write_json(directory / "result.json", result)
    return result


def aggregate(root, rows, seeds, smoke=False):
    keys = [r["training_seed"] for r in rows]
    if len(keys) != len(set(keys)) or not set(keys).issubset(seeds):
        raise ValueError("Unexpected or repeated training seed")
    rows = sorted(rows, key=lambda r: r["training_seed"])
    metrics = {m: seed_stats([r[m] for r in rows]) for m in METRICS}
    comparisons = {}
    for label in ("full", "none", "fixed_5"):
        comparisons[label] = {m: seed_stats([r[m] - r["comparisons"][label]["baseline_metrics"][m] for r in rows]) for m in METRICS}
    complete = set(keys) == set(seeds)
    overall_pass = targets_met(metrics["success_rate"]["mean"], metrics["attempted_links"]["mean"]) if rows else False
    result = {"complete": complete, "smoke_only": smoke, "training_seeds": seeds,
              "completed_training_seeds": keys, "metrics": metrics, "per_seed": rows,
              "targets": {"success_strictly_above": .9, "mean_packets_strictly_below": 120.},
              "aggregate_targets_met": overall_pass, "seeds_meeting_both_heldout_targets": sum(r["heldout_targets_met"] for r in rows),
              "paired_seed_differences_vs_baselines": comparisons,
              "environment": {"min_agent_distance": .4, "min_landmark_distance": .4, "loss": 0.,
                              "horizon": 100, "terminate_on_collision": False, "task_reward_version": "legacy-distance-v0"},
              "scope": "Per-seed warm starts from matched full baselines, with additional adaptive training. Not an equal-compute algorithm comparison. Targets are point estimates and average packets, not a hard per-episode cap."}
    write_json(root / "summary.json", result)
    fields = ["training_seed", *METRICS, "heldout_targets_met", "confirmation_targets_met",
              "adaptive_training_steps", "training_stop_reason", "max_packets_in_episode", "episodes_at_or_above_120_packets"]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    lines = ["# Adaptive scheduler: 15 seeds, zero communication loss" + (" (SMOKE ONLY)" if smoke else ""), "",
             f"Status: {'complete' if complete else 'in progress'}; {len(rows)}/{len(seeds)} seeds evaluated.", "",
             "Targets: final-step success strictly above 90% AND mean attempted directed packets strictly below 120 per 100-step episode. Both targets use point estimates. Traffic is an episode average, not a hard cap on individual episodes.", "",
             "Three agents, three static landmarks, zero communication loss, pairwise radial agent-start and landmark separation >=0.4, sensing radius 0.6 and coverage radius 0.1. Episodes run all 100 steps; collisions are recorded without terminating episodes. Collision step rate is the fraction of steps with any overlapping agent pair. This matches the corrected baseline workflow.", "",
             "Each seed warm-starts its own full-communication baseline actor (8,960,000 baseline training steps); movement and transmission then learn jointly with a fresh critic and optimizer. Additional adaptive steps are reported separately. Shared base task-return scoring matches the baselines (legacy-distance-v0); adaptive shaping rewards differ. This is not a comparison at equal training compute.", "",
             "Validation uses 200 batched scenarios (seed 100000). Passing validation triggers separate 200-episode original-environment confirmation (500000-500199). Training stops on confirmed targets or its step ceiling. Final frozen-policy evaluation uses the baseline-matched 200 scenarios (400000-400199); held-out results never select or tune checkpoints. A confirmation pass does not guarantee a held-out pass.", "",
             "| Completed seeds | Success mean | Success seed SD | Mean packets/episode | Collision step rate | Both targets on held-out |",
             "| --- | --- | --- | --- | --- | --- |"]
    if rows:
        sd = metrics["success_rate"]["sample_sd"]
        lines.append(f"| {len(rows)} | {metrics['success_rate']['mean']:.2%} | {f'{sd:.2%}' if sd is not None else 'pending'} | {metrics['attempted_links']['mean']:.2f} | {metrics['collision_step_rate']['mean']:.2%} | {result['seeds_meeting_both_heldout_targets']}/{len(rows)} |")
        lines += ["", f"Aggregate point-estimate targets met: **{overall_pass}** ({'final' if complete else 'provisional'}).", "",
                  "| Metric | Mean | Seed SD | 95% seed bootstrap interval |", "| --- | --- | --- | --- |"]
        for m in METRICS:
            s = metrics[m]
            ci = s.get("seed_bootstrap_95pct")
            lines.append(f"| {m} | {s['mean']:.6f} | {s['sample_sd'] if s['sample_sd'] is not None else 'pending'} | {ci if ci else 'pending'} |")
    else:
        lines.append("| 0 | pending | pending | pending | pending | pending |")
    lines += ["", "## Individual training seeds", "", "| Seed | Held-out success (95% Wilson) | Coverage | Packets/episode | Collision step rate | Adaptive steps | Confirmation pass | Held-out pass |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lo, hi = r["success_95pct_wilson"]
        lines.append(f"| {r['training_seed']} | {r['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {r['final_coverage']:.1%} | {r['attempted_links']:.2f} | {r['collision_step_rate']:.2%} | {r['adaptive_training_steps']:,} | {r['confirmation_targets_met']} | {r['heldout_targets_met']} |")
    lines += ["", "## Paired comparisons with completed matching baseline seeds", "", "| Baseline | Success change (pp) | Packet change/episode | Collision step rate change (pp) |", "| --- | --- | --- | --- |"]
    for label, differences in comparisons.items():
        if rows:
            lines.append(f"| {label} | {100*differences['success_rate']['mean']:+.2f} | {differences['attempted_links']['mean']:+.2f} | {100*differences['collision_step_rate']['mean']:+.2f} |")
    lines += ["", "Across-seed summaries give each training seed equal weight. Sample SDs and 95% percentile bootstrap intervals use training seeds as independent units (20,000 resamples; analysis seed 20261110). Per-policy Wilson intervals describe evaluation episodes only. Partial results are provisional; failed/budget-exhausted target attempts are not hidden or excluded from final evaluation.", "",
              "Artifacts: per-seed.csv, summary.json, seed-N/training/learning.jsonl, seed-N/training/config.json, checkpoint/initializer hashes, confirmations, frozen evaluation/episodes.jsonl, selection.json and result.json. The live progress.html also opens directly from disk and refreshes every five seconds.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if not smoke:
        shutil.copyfile(root / "report.md", REPO / "docs/adaptive-15-seeds-results.md")


def progress_snapshot(root, seeds, jobs, steps, started):
    # Reporting contention must never unwind the worker executor. If a file is
    # persistently locked, keep training and retry reporting on the next cycle.
    try:
        _progress_snapshot(root, seeds, jobs, steps, started)
        return True
    except OSError as error:
        print(f"Progress update deferred; training continues: {error}", file=sys.stderr, flush=True)
        return False


def _progress_snapshot(root, seeds, jobs, steps, started):
    from serve_progress import tail_records, episode_count
    runs = []
    for seed in seeds:
        directory = root / f"seed-{seed}"
        training = read_json(directory / "training/status.json", {})
        tail = tail_records(directory / "training/learning.jsonl")
        last = (tail or [{}])[-1]
        validation = next((r["validation"] for r in reversed(tail) if r.get("validation")), training.get("validation"))
        result = read_json(directory / "result.json")
        state = jobs[str(seed)]
        log = directory / "training/learning.jsonl"
        stamp = log.stat().st_mtime if log.exists() else None
        runs.append({"seed": seed, "state": state, "steps": last.get("steps", training.get("steps", 0)),
                     "step_limit": steps, "validation": validation, "confirmation": last.get("confirmation"),
                     "training_stop_reason": training.get("state"), "training_target_met": training.get("target_met", False),
                     "episodes": episode_count(directory / "evaluation/episodes.jsonl"), "result": result,
                     "updated_unix": stamp, "last_training_record": last})
    snapshot = {"generated_unix": time.time(), "started_unix": started, "runs": runs,
                "completed_seeds": sum(r["state"] == "complete" for r in runs),
                "target_seeds": sum(bool(r["result"] and r["result"]["heldout_targets_met"]) for r in runs)}
    write_json(root / "progress.json", snapshot)
    template = (root / "dashboard-template.html").read_text(encoding="utf-8")
    page = template.replace("__PROGRESS_DATA__", json.dumps(snapshot, allow_nan=False).replace("<", "\\u003c"))
    atomic_text(root / "progress.html", page)


def recover_reports(root):
    """Reverify completed policies and restore reports without training again."""
    manifest = read_json(root / "manifest.json")
    if not manifest:
        raise ValueError("Recovery requires an existing adaptive study manifest")
    seeds = manifest["training_seeds"]
    if not all((root / f"seed-{seed}/result.json").exists() for seed in seeds):
        raise ValueError("Some seed evaluations are unfinished; report recovery does not launch or replace training")
    sys.path.insert(0, str(root / "source"))
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    import torch
    torch.set_num_threads(1)
    old_status = read_json(root / "status.json", {})
    archive = root / "reporting-recovery"
    archive.mkdir(exist_ok=True)
    for name in ("status.json", "summary.json", "progress.json", "progress.html", "report.md", "per-seed.csv"):
        if (root / name).exists() and not (archive / name).exists():
            shutil.copyfile(root / name, archive / name)
    rows = []
    for seed in seeds:
        rows.append(evaluate_and_verify(root / f"seed-{seed}", seed,
                    Path(manifest["baseline_study"]), manifest["initializers"][str(seed)], manifest["episodes_per_seed"]))
    aggregate(root, rows, seeds, manifest["smoke_only"])
    episodes = []
    for seed in seeds:
        path = root / f"seed-{seed}/evaluation/episodes.jsonl"
        for line in path.read_text().splitlines():
            episodes.append(json.dumps({"training_seed": seed, **json.loads(line)}) + "\n")
    atomic_text(root / "episodes.jsonl", "".join(episodes))
    jobs = {str(seed): "complete" for seed in seeds}
    started = old_status.get("started_unix", time.time())
    write_json(root / "status.json", {"state": "complete", "jobs": jobs,
               "started_unix": started, "updated_unix": time.time(),
               "completed_seeds": len(rows), "expected_seeds": len(seeds),
               "reporting_recovered": True, "recovery_reason": "Windows file lock interrupted reporting; saved policy evaluations independently reverified"})
    shutil.copyfile(REPO / "scripts/adaptive-progress-template.html", root / "dashboard-template.html")
    if not progress_snapshot(root, seeds, jobs, manifest["max_additional_steps_per_seed"], started):
        raise RuntimeError("Evaluations and reports recovered, but progress page remains locked")
    write_json(root / "reporting-recovery/recovery.json", {"recovered_unix": time.time(),
               "verified_seeds": seeds, "episodes": len(episodes), "training_restarted": False})
    print(f"Recovered {len(rows)} verified seeds and {len(episodes)} episodes without retraining", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100_000_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recover-reports", action="store_true", help="Reverify saved results and rebuild reports; never starts training")
    cli = parser.parse_args()
    if cli.recover_reports:
        recover_reports(cli.output.resolve())
        return
    if cli.workers < 1 or cli.steps < 6400 or cli.steps % 6400:
        parser.error("positive workers and steps divisible by 6400 are required")
    baseline, root = cli.baseline.resolve(), cli.output.resolve()
    seeds = [42] if cli.smoke else list(range(42, 57))
    steps, episodes = (12800, 2) if cli.smoke else (cli.steps, 200)
    source_manifest = read_json(baseline / "manifest.json", {})
    # Preserve the baseline simulator and task-return implementation verbatim.
    for name, digest in source_manifest["source_sha256"].items():
        if hashlib.sha256((baseline / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baseline source changed: {name}")
    sys.path.insert(0, str(baseline / "source"))
    import torch
    torch.set_num_threads(1)
    initializers = preflight(baseline, seeds)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "source"
    shutil.copytree(baseline / "source", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(Path(__file__), root / "runner.py")
    shutil.copyfile(REPO / "scripts/adaptive-progress-template.html", root / "dashboard-template.html")
    write_json(root / "manifest.json", {"training_seeds": seeds, "baseline_study": str(baseline),
        "workers": cli.workers, "max_additional_steps_per_seed": steps, "episodes_per_seed": episodes,
        "smoke_only": cli.smoke, "initializers": initializers, "min_agent_distance": .4,
        "min_landmark_distance": .4, "loss": 0., "horizon": 100, "terminate_on_collision": False,
        "source_sha256": source_manifest["source_sha256"], "target_success_strictly_above": .9,
        "target_mean_packets_strictly_below": 120., "reward": "Existing adaptive coverage/traffic dual shaping; baseline-matched legacy task-return scoring",
        "validation_seed": 100000, "confirmation_seed_start": 500000, "heldout_seed_start": 400000})
    env = {**os.environ, "PYTHONPATH": str(snapshot), "OMP_NUM_THREADS": "1"}
    jobs = {str(seed): "queued" for seed in seeds}
    started = time.time()
    rows = []

    def child(command, log):
        command = list(map(str, command))
        with log.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            subprocess.run([sys.executable, "-u", *map(str, command)], cwd=REPO, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def run(seed):
        directory = root / f"seed-{seed}"
        directory.mkdir()
        jobs[str(seed)] = "training"
        initializer = initializers[seed]
        child(training_command(seed, directory, initializer["path"], steps, episodes), directory / "training.console.log")
        status = read_json(directory / "training/status.json")
        if status["state"] == "user_stop":
            raise RuntimeError("Training was stopped by a STOP file; held-out evaluation not performed")
        checkpoint = "training/target.pt" if status["target_met"] else "training/best.pt"
        write_json(directory / "selection.json", {"checkpoint": checkpoint,
            "rule": "Confirmation-qualified target checkpoint, otherwise validation-selected best; held-out results not used",
            "training_stop_reason": status["state"]})
        jobs[str(seed)] = "evaluating"
        child(["-m", "limited_comm.evaluate", directory / checkpoint, "--trained-schedule-only", "--loss", "0",
               "--min-agent-distance", ".4", "--episodes", str(episodes), "--seed", "400000",
               "--output", directory / "evaluation"], directory / "evaluation.console.log")
        jobs[str(seed)] = "verifying"
        return evaluate_and_verify(directory, seed, baseline, initializer, episodes)

    aggregate(root, rows, seeds, cli.smoke)
    progress_snapshot(root, seeds, jobs, steps, started)
    with ThreadPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(run, seed): seed for seed in seeds}
        pending = set(futures)
        while pending:
            for future in list(pending):
                if not future.done():
                    continue
                seed = futures[future]
                pending.remove(future)
                try:
                    rows.append(future.result())
                    jobs[str(seed)] = "complete"
                except Exception as error:
                    jobs[str(seed)] = "failed"
                    write_json(root / f"seed-{seed}/failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
                aggregate(root, rows, seeds, cli.smoke)
            write_json(root / "status.json", {"state": "running" if pending else "complete" if len(rows) == len(seeds) else "failed",
                       "jobs": dict(jobs), "started_unix": started, "updated_unix": time.time(),
                       "completed_seeds": len(rows), "expected_seeds": len(seeds)})
            progress_snapshot(root, seeds, jobs, steps, started)
            if pending:
                time.sleep(5)
    # Concatenate verified episode records, retaining the independent training seed.
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: r["training_seed"]):
            path = root / f"seed-{row['training_seed']}/evaluation/episodes.jsonl"
            for line in path.read_text().splitlines():
                stream.write(json.dumps({"training_seed": row["training_seed"], **json.loads(line)}) + "\n")
    if len(rows) != len(seeds):
        raise SystemExit("Some adaptive seeds failed; inspect per-seed failure.json and logs")


if __name__ == "__main__":
    main()
