"""Frozen N=3-trained actor transfer to 5, 10 and 20-agent SimpleSpread."""
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
OUTPUT = REPO / "results/scheduler-team-sizes-zero-loss-20261111"
METHODS = {"adaptive": "adaptive", "full": "full", "fixed_5": "fixed", "none": "none"}
TEAMS = (5, 10, 20)
METRICS = ("success_rate", "final_coverage", "task_return", "task_return_per_agent",
           "collision_step_rate", "collision_episode_fraction", "collision_pairs_per_step",
           "attempted_links", "delivered_links", "packet_ratio_fixed_denominator")


def prepare_layouts(root, episodes):
    import numpy as np
    from limited_comm.env import CommunicationSpread
    from limited_comm.layout import minimum_separation
    layouts = {}
    for n in TEAMS:
        env = CommunicationSpread(n=n, layout_sampler="sequential", min_agent_distance=.4, min_landmark_distance=.4)
        try:
            for seed in range(400000, 400000 + episodes):
                env.reset(seed=seed)
                positions = np.asarray([*[a.state.p_pos for a in env.world.agents],
                                        *[l.state.p_pos for l in env.world.landmarks]], dtype="<f8")
                assert minimum_separation(positions[:n]) >= .4 and minimum_separation(positions[n:]) >= .4
                assert (np.abs(positions) <= 1).all() and env.diagnostics()["collision_pairs"] == 0
                layouts[f"{n}/{seed}"] = {"sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                    "agents": positions[:n].tolist(), "landmarks": positions[n:].tolist()}
        finally:
            env.close()
    write_json(root / "layouts.json", layouts)
    return len(layouts)


def worker(root_string, n, seed):
    root = Path(root_string)
    manifest = read_json(root / "manifest.json")
    layouts = read_json(root / "layouts.json")
    sys.path.insert(0, str(root / "source"))
    import numpy as np
    import torch
    from limited_comm.evaluate import evaluate_episode, summarize
    from limited_comm.mappo import load_policy
    torch.set_num_threads(1)
    directory = root / f"n-{n}/seed-{seed}"
    directory.mkdir(parents=True)
    summaries, done = [], 0
    start = time.time()
    with (directory / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for method, schedule in METHODS.items():
            info = manifest["checkpoints"][str(seed)][method]
            path = Path(info["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
            model, checkpoint = load_policy(path)
            config = checkpoint["config"]
            assert config["seed"] == seed and config["training_schedule"] == schedule and config["loss"] == 0
            assert config["min_agent_distance"] == config["min_landmark_distance"] == .4
            assert model.spec["obs_dim"] == 28 and not config.get("terminate_on_collision", False)
            records = []
            for scenario in range(400000, 400000 + manifest["episodes_per_condition"]):
                row = {"training_seed": seed, "condition": method, **evaluate_episode(model, scenario,
                    schedule, period=5, loss=0., horizon=100, radius=.6,
                    min_agent_distance=.4, min_landmark_distance=.4, n=n, layout_sampler="sequential")}
                assert row["initial_layout_sha256"] == layouts[f"{n}/{scenario}"]["sha256"]
                assert row["actual_initial_agent_separation"] >= .4 and row["actual_min_separation"] >= .4
                assert not row["initial_collision"] and row["active_steps"] == 100
                assert row["attempted_links"] == row["delivered_links"]
                assert row["task_reward_version"] == "legacy-distance-v0"
                if method != "adaptive":
                    assert row["attempted_links"] == n * (n - 1) * {"full": 100, "fixed_5": 20, "none": 0}[method]
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                done += 1
                if done % 5 == 0 or scenario == 400000 + manifest["episodes_per_condition"] - 1:
                    write_json(directory / "status.json", {"state": "running", "completed": done,
                        "total": manifest["episodes_per_condition"] * 4, "agents": n, "training_seed": seed,
                        "condition": method, "updated_unix": time.time(), "seconds": time.time() - start})
            summary = summarize(records)
            summary.update(task_return_per_agent=summary["task_return"] / n,
                collision_episode_fraction=float(np.mean([r["any_collision"] for r in records])),
                collision_pairs_per_step=float(np.mean([r["collision_pairs"] / 100 for r in records])),
                packet_ratio_fixed_denominator=float(np.mean([r["packet_ratio_fixed_denominator"] for r in records])))
            summaries.append({"agents": n, "landmarks": n, "training_seed": seed, "condition": method, **summary})
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
            write_json(directory / "summary.json", summaries)
    write_json(directory / "status.json", {"state": "complete", "completed": done, "total": done,
                                           "agents": n, "training_seed": seed, "condition": "all",
                                           "updated_unix": time.time(), "seconds": time.time() - start})
    return summaries


def aggregate(root, rows, seeds, smoke=False):
    keys = [(r["agents"], r["training_seed"], r["condition"]) for r in rows]
    assert len(keys) == len(set(keys))
    conditions = []
    for n in TEAMS:
        for method in METHODS:
            subset = [r for r in rows if r["agents"] == n and r["condition"] == method]
            conditions.append({"agents": n, "landmarks": n, "condition": method, "training_seeds": len(subset),
                               "metrics": {m: seed_stats([r[m] for r in subset]) for m in METRICS}})
    result = {"complete": len(rows) == len(seeds) * 12, "smoke_only": smoke,
              "training_seeds": seeds, "conditions": conditions, "per_seed": rows,
              "scope": "Frozen actors trained at N=3 evaluated without retraining at larger N. Sequential reset sampler obeys original bounds and separation, but differs from whole-layout rejection used at N=3."}
    write_json(root / "summary.json", result)
    fields = ["agents", "landmarks", "training_seed", "condition", "episodes", *METRICS]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in sorted(rows, key=lambda r: (r["agents"], r["training_seed"], r["condition"])))
    lines = ["# Frozen communication policies: larger-team tests at zero loss" + (" (SMOKE ONLY)" if smoke else ""), "",
        f"Status: {'complete' if result['complete'] else 'in progress'}; {len(rows)}/{len(seeds)*12} seed-policy-team conditions available.", "",
        "Policies: adaptive, full, fixed_5 and separately trained none. The selected N=3 checkpoints for training seeds 42-56 are frozen and evaluated at N=5, 10 and 20, with the same number of landmarks. There is no larger-team retraining or checkpoint reselection. The shared decentralized actors still receive 28 local values: self position/velocity, up to three visible peers and landmarks, and pooled inbox. The centralized N=3 critic is not used for inference.", "",
        "Zero packet loss; sensing radius 0.6, coverage radius 0.1, 100-step horizon, collision termination disabled. Both agent starts and landmark centers obey pairwise radial separation >=0.4 in the original [-1,1] square. Landmarks are static. Success requires simultaneous coverage of all N landmarks at the final step. Collision step rate counts steps with any overlapping agent pair; collision-pair counts per step are also saved because step rates may saturate for large teams.", "",
        "Reset sampling: sequential rejection with bounded retries and whole-layout restarts, followed by label permutation. This is used consistently for all three requested team sizes. It preserves the bounds and minimum-distance rules but changes the reset distribution from the historical N=3 whole-configuration rejection sampler, which is impractical at N=20. No held-out scenario seeds are dropped, replaced or filtered. layouts.json records and hashes all initial layouts; every model and training seed is checked against them.", "",
        "Each condition uses 200 matched scenarios, seeds 400000-400199; scenario labels pair policies within a team size, not physical layouts across different team sizes. Base task-return scoring is the baseline-matched legacy-distance-v0: -0.5 * sum of nearest-agent landmark distances - overlapping pairs / N per step. Raw returns scale with team size; return per agent is saved as an additional diagnostic, not a claim of identical task difficulty.", "",
        "Performance metrics and collision-episode fractions below are mean [95% seed bootstrap interval], using equally weighted training seeds and 20,000 percentile-bootstrap resamples. Attempted packets, delivered packets and full-traffic ratios show means only. Partial results are provisional. These tests measure transfer of the trained N=3 policies, not performance of policies trained for each larger environment.", "",
        "| Agents / landmarks | Policy | Seeds | Success | Coverage | Task return | Collision step rate |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    def cell(stat, percent=False):
        def fmt(value):
            return f"{value:.2%}" if percent else f"{value:.2f}"
        ci = stat.get("seed_bootstrap_95pct")
        return fmt(stat["mean"]) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else " [pending]")
    for c in conditions:
        if c["training_seeds"]:
            m = c["metrics"]
            lines.append(f"| {c['agents']}/{c['landmarks']} | {c['condition']} | {c['training_seeds']} | {cell(m['success_rate'],True)} | {cell(m['final_coverage'],True)} | {cell(m['task_return'])} | {cell(m['collision_step_rate'],True)} |")
    lines += ["", "| Agents / landmarks | Policy | Seeds | Attempted packets/episode | Delivered packets/episode | Full-traffic ratio | Episodes with collisions |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in conditions:
        if c["training_seeds"]:
            m = c["metrics"]
            lines.append(f"| {c['agents']}/{c['landmarks']} | {c['condition']} | {c['training_seeds']} | {m['attempted_links']['mean']:.2f} | {m['delivered_links']['mean']:.2f} | {m['packet_ratio_fixed_denominator']['mean']:.2%} | {cell(m['collision_episode_fraction'],True)} |")
    lines += ["", "Full traffic is N*(N-1)*100 directed packets per episode: 2,000 at N=5, 9,000 at N=10, and 38,000 at N=20. Fixed_5 attempts one-fifth of these (400 / 1,800 / 7,600); none attempts zero. Attempted and delivered counts must match at zero loss.", "",
        "Artifacts: per-seed.csv, summary.json, n-N/seed-S/episodes.jsonl, n-N/seed-S/summary.json, layouts.json, manifest.json, status.json and progress.html. On completion, episodes.jsonl concatenates the complete original-environment evaluation records. Existing N=3 baseline and packet-loss artifacts remain separate.", ""]
    atomic_text(root / "report.md", "\n".join(lines))
    if not smoke:
        atomic_text(REPO / "docs/scheduler-team-size-zero-loss-results.md", "\n".join(lines))


def progress(root, total, state, errors):
    statuses = [read_json(p, {}) for p in root.glob("n-*/seed-*/status.json")]
    completed = sum(s.get("completed", 0) for s in statuses)
    data = {"state": state, "completed_episodes": completed, "total_episodes": total,
            "completed_team_seed_jobs": sum(s.get("state") == "complete" for s in statuses),
            "updated_unix": time.time(), "errors": errors, "jobs": statuses}
    write_json(root / "status.json", data)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>Larger-team evaluation</title>
<style>body{{max-width:1100px;padding:25px;margin:auto;background:#f5f6f9;color:#203047;font:14px system-ui}}progress{{width:100%;height:25px;accent-color:#5366ce}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#5366ce}}</style>
<h1>Frozen policies: larger-team tests</h1><p>5/5, 10/10 and 20/20 agents/landmarks · zero loss · 0.4-unit pairwise radial spacing</p><progress value="{completed}" max="{total}"></progress><p>{completed:,} / {total:,} episodes · {state} · {data['completed_team_seed_jobs']} completed team/seed jobs</p><p>Updated {time.strftime('%H:%M:%S UTC',time.gmtime())}. This page refreshes every five seconds.</p><p><a href="report.md">Comparison report</a> · <a href="per-seed.csv">Per-seed results</a></p><p>Existing N=3-trained actors are frozen. Larger resets use documented sequential rejection; no spacing threshold is relaxed.</p><table><tr><th>Agents/landmarks</th><th>Training seed</th><th>Policy</th><th>Episodes</th><th>State</th></tr>'''
    for s in sorted(statuses, key=lambda s: (s.get("agents", 0), s.get("training_seed", 0))):
        page += f"<tr><td>{s.get('agents','—')}</td><td>{s.get('training_seed','—')}</td><td>{s.get('condition','—')}</td><td>{s.get('completed',0)} / {s.get('total',0)}</td><td>{s.get('state','—')}</td></tr>"
    page += "</table></html>"
    try:
        atomic_text(root / "progress.html", page)
    except OSError as error:
        print(f"Display update deferred: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error("workers must be positive")
    root = cli.output.resolve()
    seeds = [42] if cli.smoke else list(range(42, 57))
    episodes = 2 if cli.smoke else 200
    assert read_json(BASELINE / "summary.json")["complete"] and read_json(ADAPTIVE / "summary.json")["complete"]
    checkpoints = {}
    for seed in seeds:
        checkpoints[seed] = {}
        for method in METHODS:
            directory = ADAPTIVE / f"seed-{seed}" if method == "adaptive" else BASELINE / f"seed-{seed}" / method
            path = directory / read_json(directory / "selection.json")["checkpoint"] if method == "adaptive" else directory / "best.pt"
            expected = read_json(directory / "result.json")["checkpoint_sha256"] if method == "adaptive" else read_json(directory / "selection.json")["sha256"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            checkpoints[seed][method] = {"path": str(path), "sha256": expected}
    root.mkdir(parents=True, exist_ok=False)
    source_manifest = read_json(BASELINE / "manifest.json")
    shutil.copytree(BASELINE / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    for name, digest in source_manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    for name in ("layout.py", "env.py", "evaluate.py", "landmark_channel.py"):
        shutil.copyfile(REPO / "src/limited_comm" / name, root / "source/limited_comm" / name)
    shutil.copyfile(Path(__file__), root / "runner.py")
    write_json(root / "manifest.json", {"training_seeds": seeds, "team_sizes": TEAMS,
        "episodes_per_condition": episodes, "loss": 0., "checkpoints": checkpoints, "smoke_only": cli.smoke,
        "min_agent_distance": .4, "min_landmark_distance": .4, "reset_bounds": [-1, 1], "layout_sampler": "sequential",
        "horizon": 100, "terminate_on_collision": False, "neighbors": 3,
        "original_checkpoint_team_size": 3, "retraining": False, "critic_used_for_inference": False,
        "source_sha256": {p.relative_to(root / "source").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "source").rglob("*.py")},
        "changes_from_baseline_source": ["layout.py: explicit sequential rejection option", "env.py: explicit reset sampler option", "evaluate.py: team-size and packet-denominator generalization", "landmark_channel.py: optional protocol dependency; legacy evaluation remains selected"],
        "baseline_source_sha256": source_manifest["source_sha256"]})
    sys.path.insert(0, str(root / "source"))
    prepare_layouts(root, episodes)
    rows, errors = [], {}
    aggregate(root, rows, seeds, cli.smoke)
    total = len(seeds) * 12 * episodes
    # Start expensive N=20 tasks first so smaller jobs fill the remaining slots.
    with ProcessPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(worker, str(root), n, seed): (n, seed) for n in reversed(TEAMS) for seed in seeds}
        pending = set(futures)
        while pending:
            for future in list(pending):
                if future.done():
                    pending.remove(future)
                    try:
                        rows.extend(future.result())
                    except Exception:
                        errors[str(futures[future])] = traceback.format_exc()
                    aggregate(root, rows, seeds, cli.smoke)
            progress(root, total, "running" if pending else "failed" if errors else "complete", errors)
            if pending:
                time.sleep(5)
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for n in TEAMS:
            for seed in seeds:
                p = root / f"n-{n}/seed-{seed}/episodes.jsonl"
                if str((n, seed)) not in errors:
                    stream.write(p.read_text())
    if errors:
        raise SystemExit("Some larger-team evaluations failed; inspect status.json")


if __name__ == "__main__":
    main()
