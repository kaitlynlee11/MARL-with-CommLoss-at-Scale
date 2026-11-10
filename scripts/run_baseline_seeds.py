"""Train all 15 baseline seeds with 0.4-unit agent and landmark spacing.

Runs an immutable source snapshot with the historical reward and radial
agent starts at least 0.4 units apart. All policies start from scratch;
historical measurements are never included in the corrected study.
Use --smoke for a separate, short end-to-end verification (never final results).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LABELS = {"full": "full", "none": "none", "fixed_5": "fixed"}
METRICS = ("success_rate", "final_coverage", "task_return", "attempted_links", "collision_step_rate")
LEGACY_REWARD = '''"""Historical baseline reward, before November 5, 2026."""
import numpy as np
TASK_REWARD_VERSION = "legacy-distance-v0"
def task_reward(distance_sum, collision_pairs, full_coverage, n):
    return -0.5 * np.asarray(distance_sum) - np.asarray(collision_pairs) / n
'''


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def historical_rows():
    """Parse the actual published table; do not invent missing episode records."""
    rows = []
    for line in (REPO / "docs/spaced-navigation-results.md").read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells and cells[0] in LABELS:
            rows.append(dict(training_seed=42, condition=cells[0], episodes=200,
                             success_rate=float(cells[1].split("%")[0]) / 100,
                             final_coverage=float(cells[2].strip("%")) / 100,
                             task_return=float(cells[3]), attempted_links=float(cells[4]),
                             collision_step_rate=float(cells[5].strip("%")) / 100,
                             provenance="published rounded aggregate; original artifacts unavailable"))
    if {r["condition"] for r in rows} != set(LABELS) or len(rows) != 3:
        raise ValueError("Cannot identify exactly three published baseline rows")
    return rows


def report(root, rows, expected, smoke=False, measured_only=False, document_path=None, min_agent_distance=.4):
    keys = [(r["training_seed"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate training seed/condition; cannot count it as another replicate")
    rows = sorted(rows, key=lambda r: (r["training_seed"], list(LABELS).index(r["condition"])))
    fields = ["training_seed", "condition", "episodes", *METRICS, "provenance"]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in rows)
    summaries = {}
    for label in LABELS:
        subset = [r for r in rows if r["condition"] == label]
        metrics = {}
        for metric in METRICS:
            values = np.array([r[metric] for r in subset])
            stats = {"mean": float(values.mean()) if len(values) else None,
                     "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None}
            # Seed is the independent unit. A nonparametric bootstrap avoids
            # incorrectly treating all evaluation episodes as training replicates.
            if len(values) > 1:
                rng = np.random.default_rng(20261109)
                means = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
                stats["seed_bootstrap_95pct"] = np.quantile(means, [.025, .975]).tolist()
            metrics[metric] = stats
        summaries[label] = {"training_seeds": len(subset), "metrics": metrics}
    complete = len(rows) == expected * 3 and all(s["training_seeds"] == expected for s in summaries.values())
    if measured_only and any("rounded" in r["provenance"] for r in rows):
        raise ValueError("The measured study cannot include published rounded estimates")
    provenance_note = ("All included rows are freshly measured from checkpoint evaluations. Seed 42 is rerun from scratch using the same source snapshot, training budgets and held-out scenarios as seeds 43-56. Historical rounded values are excluded."
                       if measured_only else
                       "Seed 42 uses rounded published aggregates, without original checkpoints/episodes. Means, SDs and bootstrap intervals inherit that rounding. No episode-level records are fabricated.")
    result = {"complete": complete, "smoke_only": smoke, "expected_seeds_per_condition": expected,
              "completed_condition_runs": len(rows), "conditions": summaries, "per_seed": rows,
              "measured_only": measured_only, "min_agent_distance": min_agent_distance,
              "min_landmark_distance": .4, "task_reward_version": "legacy-distance-v0", "caveat": provenance_note}
    write_json(root / "summary.json", result)
    lines = ["# Baselines across training seeds" + (" (SMOKE ONLY)" if smoke else ""), "",
             f"Status: {'complete' if complete else 'in progress'}; {len(rows)}/{expected * 3} condition runs available.", "",
             f"Reward: -0.5 * sum(landmark nearest-agent distances) - overlapping pairs / 3; minimum pairwise radial landmark center distance 0.4, minimum pairwise radial agent starting center distance {min_agent_distance:g}, loss 0, horizon 100. Spacing applies to training, curriculum, validation, evaluation and replays. Agents may move closer after reset. Each policy is trained from scratch under its own communication schedule. Validation alone selects checkpoints. Held-out scenarios are paired across all policies (400000-400199).", "",
             provenance_note, "",
             "Means give each training seed equal weight. SD and 95% percentile bootstrap intervals (20,000 resamples, fixed analysis RNG seed) describe variation across training seeds. Partial results are provisional; no significance claims are made.", "",
             "| Condition | Seeds | Success mean | Seed SD | 95% seed bootstrap interval | Coverage mean | Task return mean | Packets/episode | Collision step rate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, summary in summaries.items():
        m = summary["metrics"]
        s = m["success_rate"]
        ci = s.get("seed_bootstrap_95pct")
        interval = f"{ci[0]:.2%} to {ci[1]:.2%}" if ci else "pending"
        sd = f"{s['sample_sd']:.2%}" if s['sample_sd'] is not None else "pending"
        if summary['training_seeds']:
            lines.append(f"| {label} | {summary['training_seeds']} | {s['mean']:.2%} | {sd} | {interval} | {m['final_coverage']['mean']:.2%} | {m['task_return']['mean']:.3f} | {m['attempted_links']['mean']:.1f} | {m['collision_step_rate']['mean']:.2%} |")
        else:
            lines.append(f"| {label} | 0 | pending | pending | pending | pending | pending | pending | pending |")
    lines += ["", "## Individual training seeds", "", "| Seed | Condition | Success | Coverage | Task return | Packets/episode | Collision step rate |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['training_seed']} | {r['condition']} | {r['success_rate']:.1%} | {r['final_coverage']:.2%} | {r['task_return']:.3f} | {r['attempted_links']:.0f} | {r['collision_step_rate']:.2%} |")
    lines += ["", "Machine-readable results: `per-seed.csv`, `summary.json`; new run artifacts: `seed-N/{full,none,fixed_5}/`. `status.json` records running/completed/failed jobs. Source and historical report copies plus hashes are saved with this study.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if not smoke:
        shutil.copyfile(root / "report.md", document_path or REPO / "docs/baseline-15-seeds-results.md")
    return complete


def verify_condition(directory, seed, label, args):
    evaluation = json.loads((directory / "evaluation/summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (directory / "evaluation/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == args.episodes
    assert [r["seed"] for r in records] == list(range(400000, 400000 + args.episodes))
    assert evaluation["training_config"]["seed"] == seed
    assert evaluation["training_config"]["training_schedule"] == LABELS[label]
    assert evaluation["training_config"]["task_reward_version"] == "legacy-distance-v0"
    assert hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest() == evaluation["checkpoint_sha256"]
    for phase, steps in [("initial", args.initial_steps), ("refine", args.initial_steps + args.refine_steps)]:
        config = json.loads((directory / phase / "config.json").read_text(encoding="utf-8"))
        assert config["seed"] == seed and config["no_early_stop"] and config["min_agent_distance"] == args.min_agent_distance
        assert config["min_landmark_distance"] == args.min_landmark_distance
        last = json.loads((directory / phase / "learning.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert last["steps"] == steps
    assert evaluation["training_config"]["min_agent_distance"] == args.min_agent_distance
    assert evaluation["min_agent_distance"] == args.min_agent_distance
    assert all(r["task_reward_version"] == "legacy-distance-v0" and r["min_agent_distance"] == args.min_agent_distance and
               r["actual_initial_agent_separation"] >= args.min_agent_distance and
               (args.min_agent_distance < .4 or not r["initial_collision"]) and
               r["actual_min_separation"] >= .4 and r["loss"] == 0 and
               r["attempted_links"] == r["delivered_links"] and
               r["attempted_links"] == {"full": 600, "none": 0, "fixed_5": 120}[label] and
               not r["terminate_on_collision"] for r in records)
    from limited_comm.evaluate import summarize
    recalculated = summarize(records)
    assert all(np.isclose(recalculated[m], evaluation["conditions"][label][m]) for m in METRICS)
    return {"training_seed": seed, "condition": label, **recalculated,
            "provenance": str(directory.relative_to(args.root))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "results/baseline-agent-spacing04-15-seeds-20261109")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error("workers must be positive")
    root = cli.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    seeds = [42] if cli.smoke else list(range(42, 57))
    snapshot = root / "source"
    shutil.copytree(REPO / "src/limited_comm", snapshot / "limited_comm", ignore=shutil.ignore_patterns("__pycache__"))
    (snapshot / "limited_comm/rewards.py").write_text(LEGACY_REWARD, encoding="utf-8")
    train_source = snapshot / "limited_comm/train.py"
    train_source.write_text(train_source.read_text(encoding="utf-8").replace(
        '"coverage-gated baseline"', '"legacy ungated distance baseline"'), encoding="utf-8")
    shutil.copyfile(REPO / "docs/spaced-navigation-results.md", root / "historical-seed42-report.md")
    shutil.copyfile(Path(__file__), root / "runner.py")
    manifest = {"training_seeds": seeds, "new_training_seeds": seeds, "workers": cli.workers,
                "min_agent_distance": .4, "min_landmark_distance": .4,
                "initial_steps": 6400 if cli.smoke else 8320000,
                "refine_steps": 6400 if cli.smoke else 640000,
                "evaluation_episodes_per_policy": 2 if cli.smoke else 200,
                "smoke_only": cli.smoke, "source_sha256": {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*.py")},
                "historical_report_sha256": hashlib.sha256((root / "historical-seed42-report.md").read_bytes()).hexdigest(),
                "protocol": "All seeds trained from scratch with pairwise radial agent-start and landmark separation >=0.4. Legacy distance reward and all other baseline settings retained to isolate the spacing change. No historical values pooled."}
    write_json(root / "manifest.json", manifest)
    sys.path.insert(0, str(snapshot))
    os.environ["PYTHONPATH"] = str(snapshot)
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from limited_comm.study import train_condition
    torch.set_num_threads(1)
    args = SimpleNamespace(root=root, initial_steps=6400 if cli.smoke else 8320000,
                           refine_steps=6400 if cli.smoke else 640000,
                           episodes=2 if cli.smoke else 200, curriculum_steps=1000000,
                           min_landmark_distance=.4, min_agent_distance=.4, test_seed=400000)
    rows = []
    expected = len(seeds)
    status = {"state": "running", "started_unix": time.time(), "jobs": {f"{seed}/{label}": "pending" for seed in seeds for label in LABELS}}
    write_json(root / "status.json", status)
    def refresh_report():
        report(root, rows, expected, cli.smoke, measured_only=True, min_agent_distance=args.min_agent_distance)
        if not cli.smoke:
            shutil.copyfile(root / "report.md", REPO / "docs/navigation-baseline-15-seeds-results.md")
    refresh_report()

    def run(seed, label):
        seed_root = root / f"seed-{seed}"
        seed_root.mkdir(exist_ok=True)
        settings = SimpleNamespace(**vars(args), training_seed=seed)
        train_condition(label, LABELS[label], seed_root, settings)
        return verify_condition(seed_root / label, seed, label, args)

    with ThreadPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(run, seed, label): f"{seed}/{label}" for seed in seeds for label in LABELS}
        for future in as_completed(futures):
            key = futures[future]
            try:
                rows.append(future.result())
                status["jobs"][key] = "complete"
                refresh_report()
            except Exception as error:
                status["jobs"][key] = f"failed: {error!r}"
            status["updated_unix"] = time.time()
            write_json(root / "status.json", status)
    status["state"] = "complete" if all(v == "complete" for v in status["jobs"].values()) else "failed"
    status["seconds"] = time.time() - status["started_unix"]
    write_json(root / "status.json", status)
    if status["state"] == "complete" and not cli.smoke:
        shutil.copyfile(root / "report.md", REPO / "docs/baseline-15-seeds-results.md")
    if status["state"] == "complete":
        with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda r: (r["training_seed"], r["condition"])):
                path = root / row["provenance"] / "evaluation/episodes.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    stream.write(json.dumps({"training_seed": row["training_seed"], **json.loads(line)}) + "\n")
    if status["state"] != "complete":
        raise SystemExit("Some jobs failed; inspect status.json and per-condition logs")


if __name__ == "__main__":
    main()
