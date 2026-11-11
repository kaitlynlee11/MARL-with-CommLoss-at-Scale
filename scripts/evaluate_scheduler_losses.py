"""Frozen adaptive/full/fixed_5 robustness on 15 matched seeds at 3 losses."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from run_adaptive_seeds import atomic_text, read_json, seed_stats, write_json

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "results/baseline-agent-spacing04-15-seeds-20261109"
ADAPTIVE = REPO / "results/adaptive-zero-loss-15-seeds-20261110"
OUTPUT = REPO / "results/scheduler-loss-comparison-20261110"
LOSSES = (.1, .25, .5)
METHODS = {"adaptive": "adaptive", "full": "full", "fixed_5": "fixed"}
METRICS = ("success_rate", "final_coverage", "collision_step_rate", "collision_episode_fraction",
           "collision_pairs", "attempted_links", "delivered_links", "lost_links")


def summarize_rows(rows):
    from limited_comm.evaluate import summarize
    summary = summarize(rows)
    summary.update(collision_episode_fraction=sum(r["any_collision"] for r in rows) / len(rows),
                   collision_pairs=sum(r["collision_pairs"] for r in rows) / len(rows),
                   lost_links=sum(r["attempted_links"] - r["delivered_links"] for r in rows) / len(rows))
    attempts = sum(r["attempted_links"] for r in rows)
    summary["realized_loss_fraction"] = sum(r["attempted_links"]-r["delivered_links"] for r in rows) / attempts if attempts else None
    return summary


def check_episode(row, original, method, loss):
    assert row["loss"] == loss and row["seed"] == original["seed"]
    assert row["actual_initial_agent_separation"] == original["actual_initial_agent_separation"] >= .4
    assert row["actual_min_separation"] == original["actual_min_separation"] >= .4
    assert not row["initial_collision"] and row["active_steps"] == 100 and not row["terminate_on_collision"]
    assert 0 <= row["delivered_links"] <= row["attempted_links"] <= 600
    assert row["task_reward_version"] == original["task_reward_version"] == "legacy-distance-v0"
    if method != "adaptive":
        assert row["attempted_links"] == (600 if method == "full" else 120)
    # A deterministic policy that never attempts a packet in the zero-loss
    # trajectory sees exactly the same observations under any loss draw.
    if original["attempted_links"] == 0:
        assert all(row[k] == original[k] for k in ("final_success", "final_coverage", "collision_step_rate", "task_return", "attempted_links"))


def worker(root_string, seed):
    root = Path(root_string)
    manifest = read_json(root / "manifest.json")
    sys.path.insert(0, str(root / "source"))
    import torch
    from limited_comm.mappo import load_policy
    from limited_comm.evaluate import evaluate_episode
    torch.set_num_threads(1)
    directory = root / f"seed-{seed}"
    directory.mkdir()
    completed = 0
    summaries = []
    models, originals = {}, {}
    for method in METHODS:
        info = manifest["checkpoints"][str(seed)][method]
        path = Path(info["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        model, checkpoint = load_policy(path)
        cfg = checkpoint["config"]
        assert cfg["seed"] == seed and cfg["training_schedule"] == METHODS[method]
        assert cfg["min_agent_distance"] == cfg["min_landmark_distance"] == .4
        assert cfg["loss"] == 0 and cfg["horizon"] == 100 and not cfg.get("terminate_on_collision", False)
        models[method] = model
        originals[method] = [json.loads(line) for line in Path(info["zero_loss_episodes"]).read_text().splitlines()][:manifest["episodes_per_condition"]]
    with (directory / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for loss in LOSSES:
            for method, schedule in METHODS.items():
                records = []
                for index, original in enumerate(originals[method]):
                    row = {"training_seed": seed, "condition": method, **evaluate_episode(
                        models[method], original["seed"], schedule, period=5, loss=loss,
                        horizon=100, radius=.6, min_agent_distance=.4, min_landmark_distance=.4,
                        terminate_on_collision=False)}
                    check_episode(row, original, method, loss)
                    records.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    completed += 1
                    if (index + 1) % 10 == 0 or index + 1 == len(originals[method]):
                        write_json(directory / "status.json", {"state": "running", "completed": completed,
                            "total": manifest["episodes_per_condition"] * 9, "loss": loss,
                            "condition": method, "updated_unix": time.time()})
                summary = summarize_rows(records)
                zero = summarize_rows(originals[method])
                summaries.append({"training_seed": seed, "loss": loss, "condition": method, **summary,
                    "zero_loss_metrics": {m: zero[m] for m in METRICS},
                    "difference_vs_zero_loss": {m: summary[m] - zero[m] for m in METRICS}})
    for method in METHODS:
        info = manifest["checkpoints"][str(seed)][method]
        assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
    write_json(directory / "summary.json", summaries)
    write_json(directory / "status.json", {"state": "complete", "completed": completed,
                                           "total": completed, "updated_unix": time.time()})
    return summaries


def aggregate(root, rows, expected_seeds, smoke=False):
    keys = [(r["training_seed"], r["loss"], r["condition"]) for r in rows]
    assert len(keys) == len(set(keys)), "Repeated independent condition"
    conditions = []
    for loss in LOSSES:
        for method in METHODS:
            subset = [r for r in rows if r["loss"] == loss and r["condition"] == method]
            conditions.append({"loss": loss, "condition": method, "training_seeds": len(subset),
                "metrics": {m: seed_stats([r[m] for r in subset]) for m in METRICS},
                "change_vs_zero_loss": {m: seed_stats([r["difference_vs_zero_loss"][m] for r in subset]) for m in METRICS}})
    result = {"complete": len(rows) == expected_seeds * 9, "smoke_only": smoke,
              "expected_seeds": expected_seeds, "conditions": conditions, "per_seed": rows,
              "loss_rate_awareness": "Rates are known/configured in the environment. Existing frozen actors have no rate input or acknowledgments; these are not loss-conditioned policies."}
    write_json(root / "summary.json", result)
    fields = ["training_seed", "loss", "condition", "episodes", *METRICS]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in sorted(rows, key=lambda r: (r["training_seed"], r["loss"], r["condition"])))
    lines = ["# Frozen scheduler comparison under communication loss" + (" (SMOKE ONLY)" if smoke else ""), "",
        f"Status: {'complete' if result['complete'] else 'in progress'}; {len(rows)}/{expected_seeds*9} seed-policy-loss conditions available.", "",
        "All three selected zero-loss-trained policies are frozen, with no retraining or checkpoint reselection. Training seeds 42-56; 200 matched scenarios (400000-400199) per seed, policy and loss rate. Rates are configured and known to the evaluator, but actors have no loss-rate input or acknowledgments. This evaluates robustness of the existing policies, not explicitly loss-aware policies.", "",
        "Independent Bernoulli loss applies to each attempted directed packet. All policies and loss rates use the same seed-indexed per-link/per-step random draws, including draws for silent senders. Agent starts and landmarks have minimum pairwise radial separation 0.4; horizon is 100, without collision termination. Success is simultaneous full landmark coverage at the final step. Coverage is the final fraction of landmarks covered.", "",
        "Completed packets means successfully delivered directed packets per episode. Attempted and dropped packets are recorded separately. Collision step rate is the fraction of steps with any overlapping pair; collision-episode fraction is the fraction of episodes with at least one collision. Collision-pair counts per episode are also saved.", "",
        "| Loss | Policy | Seeds | Success | Coverage | Collision step rate | Episodes with collisions | Delivered packets/episode | Attempted packets/episode |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in conditions:
        m = c["metrics"]
        if c["training_seeds"]:
            lines.append(f"| {c['loss']:.0%} | {c['condition']} | {c['training_seeds']} | {m['success_rate']['mean']:.2%} | {m['final_coverage']['mean']:.2%} | {m['collision_step_rate']['mean']:.2%} | {m['collision_episode_fraction']['mean']:.2%} | {m['delivered_links']['mean']:.2f} | {m['attempted_links']['mean']:.2f} |")
    lines += ["", "Means weight independent training seeds equally. summary.json includes sample SDs, 95% percentile bootstrap intervals across training seeds (20,000 resamples), and paired changes against each policy's original zero-loss results. Wilson intervals in per-seed summaries measure episode uncertainty only. Scenario pairing does not make evaluation episodes independent training replicates.", "",
        "One seeded channel realization is used per physical scenario. Nearly silent adaptive policies may be insensitive to loss because they do not rely on delivered packets; zero-loss and loss-tested trajectories are verified identical whenever the zero-loss trajectory attempts no packets. No general benefit of packet loss is implied by small observed improvements.", "",
        "Individual-seed results: per-seed.csv and seed-N/summary.json. Episode records: episodes.jsonl and seed-N/episodes.jsonl. Checkpoint/source hashes and settings: manifest.json. Live completion: progress.html and status.json.", ""]
    atomic_text(root / "report.md", "\n".join(lines))
    if not smoke:
        atomic_text(REPO / "docs/scheduler-loss-comparison-results.md", "\n".join(lines))


def progress(root, total, state="running", errors=None):
    completed = sum(read_json(p, {}).get("completed", 0) for p in root.glob("seed-*/status.json"))
    status = {"state": state, "completed_episodes": completed, "total_episodes": total,
              "updated_unix": time.time(), "errors": errors or {}}
    write_json(root / "status.json", status)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Scheduler packet-loss tests</title>
<style>body{{max-width:950px;margin:45px auto;padding:0 24px;background:#f5f6f9;color:#203047;font:15px system-ui}}progress{{width:100%;height:25px;accent-color:#5365cc}}a{{color:#5365cc}}pre{{white-space:pre-wrap}}</style>
<h1>Communication-loss comparison</h1><p>Adaptive · Full communication · Every 5 steps</p>
<p>10%, 25%, 50% configured loss · 15 training seeds · 0.4-unit agent and landmark spacing</p>
<progress value="{completed}" max="{total}"></progress><p>{completed:,} / {total:,} episodes · {state}</p>
<p>Updated {time.strftime('%H:%M:%S UTC', time.gmtime())}</p><p><a href="report.md">Comparison report</a> · <a href="per-seed.csv">Individual seed results</a></p>
<p>Frozen policies; no loss-rate observation is added. Collisions, delivered and attempted packets, final coverage and success are measured.</p></html>'''
    try:
        atomic_text(root / "progress.html", page)
    except OSError as error:
        print(f"Progress display update deferred: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error("workers must be positive")
    root = cli.output.resolve()
    seeds = [42, 45] if cli.smoke else list(range(42, 57))
    episodes = 2 if cli.smoke else 200
    assert read_json(ADAPTIVE / "summary.json")["complete"]
    assert read_json(BASELINE / "summary.json")["complete"]
    checkpoints = {}
    for seed in seeds:
        checkpoints[seed] = {}
        for method in METHODS:
            directory = ADAPTIVE / f"seed-{seed}" if method == "adaptive" else BASELINE / f"seed-{seed}" / method
            path = directory / read_json(directory / "selection.json")["checkpoint"] if method == "adaptive" else directory / "best.pt"
            expected = read_json(directory / "result.json")["checkpoint_sha256"] if method == "adaptive" else read_json(directory / "selection.json")["sha256"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            checkpoints[seed][method] = {"path": str(path), "sha256": expected,
                                       "zero_loss_episodes": str(directory / "evaluation/episodes.jsonl")}
    source_manifest = read_json(BASELINE / "manifest.json")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASELINE / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    for name, digest in source_manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    shutil.copyfile(Path(__file__), root / "runner.py")
    write_json(root / "manifest.json", {"training_seeds": seeds, "loss_rates": LOSSES,
        "episodes_per_condition": episodes, "checkpoints": checkpoints, "smoke_only": cli.smoke,
        "source_sha256": source_manifest["source_sha256"], "actor_loss_observation": False,
        "loss_knowledge": "Configured environment rates; frozen original 28-value actor observations",
        "min_agent_distance": .4, "min_landmark_distance": .4, "horizon": 100,
        "terminate_on_collision": False, "source_baseline": str(BASELINE), "source_adaptive": str(ADAPTIVE)})
    rows, errors = [], {}
    aggregate(root, rows, len(seeds), cli.smoke)
    total = len(seeds) * 9 * episodes
    with ProcessPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(worker, str(root), seed): seed for seed in seeds}
        pending = set(futures)
        while pending:
            for future in list(pending):
                if future.done():
                    pending.remove(future)
                    try:
                        rows.extend(future.result())
                    except Exception:
                        errors[str(futures[future])] = traceback.format_exc()
                    aggregate(root, rows, len(seeds), cli.smoke)
            progress(root, total, "running" if pending else "failed" if errors else "complete", errors)
            if pending:
                time.sleep(5)
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for seed in seeds:
            if str(seed) not in errors:
                stream.write((root / f"seed-{seed}/episodes.jsonl").read_text())
    if errors:
        raise SystemExit("Some evaluations failed; inspect status.json")


if __name__ == "__main__":
    main()
