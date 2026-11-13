"""Paired transmit-off ablation of 15 frozen adaptive policies on held-out N=3."""
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

import numpy as np

from run_adaptive_seeds import atomic_text, read_json, seed_stats, write_json

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/adaptive-zero-loss-15-seeds-20261110"
OUTPUT = REPO / "results/adaptive-communication-ablation-20261112"
METRICS = ("success_rate", "collision_step_rate", "collision_episode_fraction", "task_return",
           "final_coverage", "attempted_links", "delivered_links")


class ForceSilent:
    """Keep the exact joint-action argmax's movement; discard its transmit bit."""
    def __init__(self, model):
        self.model = model
        self.spec = model.spec
        self.suppressed_gates = 0

    def reset_memory(self):
        self.model.reset_memory()
        self.suppressed_gates = 0

    def actions(self, obs):
        moves, gates = self.model.actions(obs)
        self.suppressed_gates += int(np.asarray(gates).sum())
        return moves, np.zeros_like(gates)


def check_original(actual, saved):
    """Reproduce all saved episode fields; metadata condition is adaptive."""
    for key, expected in saved.items():
        if key == "condition":
            assert expected == "adaptive"
        elif isinstance(expected, float):
            assert np.isclose(actual[key], expected, rtol=0, atol=1e-9), (key, actual[key], expected)
        else:
            assert actual[key] == expected, (key, actual[key], expected)


def summarize_rows(rows):
    from limited_comm.evaluate import summarize
    return {**summarize(rows),
            "collision_episode_fraction": float(np.mean([r["any_collision"] for r in rows]))}


def worker(root_string, seed):
    root = Path(root_string)
    manifest = read_json(root / "manifest.json")
    sys.path.insert(0, str(root / "source"))
    import torch
    from limited_comm.evaluate import evaluate_episode
    from limited_comm.mappo import load_policy
    torch.set_num_threads(1)
    info = manifest["checkpoints"][str(seed)]
    checkpoint = Path(info["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == info["sha256"]
    model, saved_checkpoint = load_policy(checkpoint)
    config = saved_checkpoint["config"]
    assert config["seed"] == seed and config["training_schedule"] == "adaptive"
    assert model.spec["adaptive"] and model.spec["obs_dim"] == 28
    assert config["min_agent_distance"] == config["min_landmark_distance"] == .4
    assert config["loss"] == 0 and config["horizon"] == 100 and config["radius"] == .6
    assert not config.get("terminate_on_collision", False)
    saved_rows = [json.loads(line) for line in Path(info["original_episodes"]).read_text().splitlines()][:manifest["episodes_per_seed"]]
    assert [r["seed"] for r in saved_rows] == list(range(400000, 400000 + manifest["episodes_per_seed"]))
    directory = root / f"seed-{seed}"
    directory.mkdir()
    silent = ForceSilent(model)
    originals, disabled, pairs = [], [], []
    with (directory / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as stream, \
         (directory / "paired-episodes.jsonl").open("x", encoding="utf-8", buffering=1) as paired_stream:
        for index, saved in enumerate(saved_rows):
            arguments = dict(seed=saved["seed"], schedule="adaptive", period=5, loss=0., horizon=100,
                             radius=.6, min_agent_distance=.4, min_landmark_distance=.4,
                             terminate_on_collision=False)
            original = evaluate_episode(model, **arguments)
            check_original(original, saved)
            ablated = evaluate_episode(silent, **arguments)
            assert ablated["attempted_links"] == ablated["delivered_links"] == ablated["payload_bytes"] == 0
            for field in ("seed", "actual_initial_agent_separation", "actual_min_separation", "task_reward_version"):
                assert original[field] == ablated[field]
            assert ablated["active_steps"] == original["active_steps"] == 100
            assert ablated["actual_initial_agent_separation"] >= .4 and ablated["actual_min_separation"] >= .4
            assert not ablated["initial_collision"] and ablated["task_reward_version"] == "legacy-distance-v0"
            if original["attempted_links"] == 0:
                assert original == ablated, "Silencing an already-silent deterministic trajectory changed its outcomes"
            originals.append(original)
            disabled.append(ablated)
            pair = {"training_seed": seed, "scenario_seed": saved["seed"],
                    "original_success": original["final_success"], "disabled_success": ablated["final_success"],
                    "original_packets": original["attempted_links"],
                    "suppressed_gate_decisions": silent.suppressed_gates,
                    "success_changed": original["final_success"] != ablated["final_success"],
                    **{f"difference_{m}": ablated[m] - original[m] for m in
                       ("task_return", "collision_step_rate", "final_coverage", "collision_pairs")}}
            pairs.append(pair)
            paired_stream.write(json.dumps(pair, allow_nan=False) + "\n")
            for name, row in (("original", original), ("communication_disabled", ablated)):
                stream.write(json.dumps({"training_seed": seed, "condition": name, **row}, allow_nan=False) + "\n")
            if (index + 1) % 10 == 0 or index + 1 == len(saved_rows):
                write_json(directory / "status.json", {"state": "running", "training_seed": seed,
                    "completed_episodes": 2 * (index + 1), "paired_scenarios": index + 1,
                    "total_episodes": 2 * len(saved_rows), "updated_unix": time.time()})
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == info["sha256"]
    original_summary, disabled_summary = summarize_rows(originals), summarize_rows(disabled)
    result = {"training_seed": seed, "original": original_summary, "communication_disabled": disabled_summary,
              "differences_disabled_minus_original": {m: disabled_summary[m] - original_summary[m] for m in METRICS},
              "paired_scenarios": len(pairs), "original_records_reproduced": True,
              "original_communicating_episodes": sum(r["attempted_links"] > 0 for r in originals),
              "success_to_failure": sum(p["original_success"] and not p["disabled_success"] for p in pairs),
              "failure_to_success": sum(not p["original_success"] and p["disabled_success"] for p in pairs),
              "collision_step_rate_changed_episodes": sum(p["difference_collision_step_rate"] != 0 for p in pairs),
              "task_return_changed_episodes": sum(abs(p["difference_task_return"]) > 1e-9 for p in pairs)}
    write_json(directory / "summary.json", result)
    write_json(directory / "status.json", {"state": "complete", "training_seed": seed,
               "completed_episodes": 2 * len(pairs), "paired_scenarios": len(pairs),
               "total_episodes": 2 * len(pairs), "updated_unix": time.time()})
    return result


def aggregate(root, rows, seeds, smoke=False, evaluation_episodes=200):
    rows = sorted(rows, key=lambda r: r["training_seed"])
    assert len({r["training_seed"] for r in rows}) == len(rows)
    stats = {name: {m: seed_stats([r[name][m] for r in rows]) for m in METRICS}
             for name in ("original", "communication_disabled", "differences_disabled_minus_original")}
    complete = len(rows) == len(seeds)
    result = {"complete": complete, "smoke_only": smoke, "training_seeds": seeds,
              "completed_training_seeds": [r["training_seed"] for r in rows], "statistics": stats, "per_seed": rows,
              "success_to_failure": sum(r["success_to_failure"] for r in rows),
              "failure_to_success": sum(r["failure_to_success"] for r in rows),
              "original_communicating_episodes": sum(r["original_communicating_episodes"] for r in rows),
              "original_records_reproduced": all(r["original_records_reproduced"] for r in rows)}
    write_json(root / "summary.json", result)
    fields = ["training_seed", *[f"{name}_{metric}" for metric in METRICS for name in ("original", "communication_disabled", "difference")],
              "success_to_failure", "failure_to_success", "original_communicating_episodes"]
    with (root / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {"training_seed": row["training_seed"], **{k: row[k] for k in fields if k in row}}
            for m in METRICS:
                for name in ("original", "communication_disabled", "difference"):
                    record[f"{name}_{m}"] = row["differences_disabled_minus_original" if name == "difference" else name][m]
            writer.writerow(record)
    lines = ["# Adaptive-policy communication ablation" + (" (SMOKE ONLY)" if smoke else ""), "",
        f"Status: {'complete' if complete else 'in progress'}; {len(rows)}/{len(seeds)} policy seeds evaluated.", "",
        "The same selected adaptive checkpoint is evaluated normally and with every agent's transmit gate forcibly set to zero. Movement is taken from the original deterministic 10-action joint argmax; only its transmit bit is discarded. The actor and recurrent memory continue operating on the resulting local observations. No retraining, checkpoint reselection, action-probability renormalization or substitution of a separately trained none policy occurs.", "",
        f"Training seeds 42-56; {evaluation_episodes} matched held-out scenarios (400000-{400000 + evaluation_episodes - 1}) per policy seed; N=3 agents/landmarks; zero loss; radial starting-agent and landmark separation >=0.4; 100 steps without collision termination. The original frozen simulator, whole-layout sampler and legacy-distance-v0 task reward are preserved with source hashes. This is the original position/velocity-only all-to-all protocol, separate from the exploratory landmark-sharing extension. Original saved episode outcomes are preserved; each normal and disabled pair uses exactly the same scenario.", "",
        "Collision step rate is the fraction of steps with any overlapping agent pair; collision-episode fraction counts episodes with any collision. Task return is the undiscounted base task reward, excluding packet penalties and training bonuses. Higher (less negative) return is better.", "",
        "Entries are mean [95% training-seed bootstrap interval]. Differences are **disabled minus original**, bootstrapped directly from the 15 paired seed differences (20,000 percentile resamples; analysis RNG seed 20261110). Rate differences are percentage points. Partial results are provisional.", "",
        "| Metric | Original adaptive | Communication disabled | Paired difference |",
        "| --- | --- | --- | --- |"]
    def cell(s, percent=False, delta=False):
        def fmt(v):
            return f"{100*v:+.3f} pp" if percent and delta else f"{v:.3%}" if percent else f"{v:+.4f}" if delta else f"{v:.4f}"
        ci = s.get("seed_bootstrap_95pct")
        return fmt(s["mean"]) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else " [pending]")
    if rows:
        for key, label, percentage in [("success_rate", "Final success", True), ("collision_step_rate", "Collision step rate", True),
            ("collision_episode_fraction", "Episodes with collisions", True), ("task_return", "Task return", False),
            ("final_coverage", "Final coverage", True), ("attempted_links", "Attempted packets/episode", False),
            ("delivered_links", "Delivered packets/episode", False)]:
            lines.append(f"| {label} | {cell(stats['original'][key],percentage)} | {cell(stats['communication_disabled'][key],percentage)} | {cell(stats['differences_disabled_minus_original'][key],percentage,True)} |")
    lines += ["", "## Per-policy seed comparison", "",
        "| Seed | Original success | Disabled success | Original collision step rate | Disabled collision step rate | Original return | Disabled return | Original packets/episode |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        a, b = r["original"], r["communication_disabled"]
        lines.append(f"| {r['training_seed']} | {a['success_rate']:.1%} | {b['success_rate']:.1%} | {a['collision_step_rate']:.3%} | {b['collision_step_rate']:.3%} | {a['task_return']:.4f} | {b['task_return']:.4f} | {a['attempted_links']:.2f} |")
    lines += ["", f"Original episodes with any transmission: {result['original_communicating_episodes']}. Paired success-to-failure changes: {result['success_to_failure']}; failure-to-success changes: {result['failure_to_success']}.", "",
        "An original episode with zero transmissions must reproduce exactly under forced silence. More generally, movement may change after messages are removed because the recurrent actor receives different inputs; preserving the frozen movement policy does not mean forcing its original action trajectory. Suppressed gate decisions describe the raw transmit choices under that counterfactual history, not the original policy's packet count.", "",
        "Artifacts: per-seed.csv, summary.json, seed-N/episodes.jsonl, seed-N/paired-episodes.jsonl, seed-N/summary.json, manifest.json and progress.html. Combined episode and paired-record files are produced on completion. Historical results remain unchanged.", ""]
    atomic_text(root / "report.md", "\n".join(lines))
    if not smoke:
        atomic_text(REPO / "docs/adaptive-communication-ablation-results.md", "\n".join(lines))


def progress(root, total, state, errors):
    completed = sum(read_json(p, {}).get("completed_episodes", 0) for p in root.glob("seed-*/status.json"))
    write_json(root / "status.json", {"state": state, "completed_episodes": completed, "total_episodes": total,
               "updated_unix": time.time(), "errors": errors})
    try:
        atomic_text(root / "progress.html", f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>Adaptive communication ablation</title><style>body{{max-width:1000px;margin:40px auto;padding:0 24px;font:15px system-ui;color:#203047;background:#f5f6f9}}progress{{width:100%;height:25px;accent-color:#5366ce}}a{{color:#5366ce}}</style><h1>Adaptive communication: on versus forced off</h1><p>Same frozen actors · 15 trained seeds · original held-out scenarios · zero loss</p><progress value="{completed}" max="{total}"></progress><p>{completed:,} / {total:,} original-plus-ablated episodes · {state}</p><p>Updated {time.strftime('%H:%M:%S UTC',time.gmtime())}</p><p><a href="report.md">Results</a> · <a href="per-seed.csv">Paired per-seed comparison</a></p></html>''')
    except OSError as error:
        print(f"Progress display deferred: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.workers < 1:
        parser.error("workers must be positive")
    seeds = [42, 45] if cli.smoke else list(range(42, 57))
    episodes = 2 if cli.smoke else 200
    assert read_json(SOURCE / "summary.json")["complete"]
    source_manifest = read_json(SOURCE / "manifest.json")
    checkpoints = {}
    for seed in seeds:
        directory = SOURCE / f"seed-{seed}"
        path = directory / read_json(directory / "selection.json")["checkpoint"]
        expected = read_json(directory / "result.json")["checkpoint_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        checkpoints[seed] = {"path": str(path), "sha256": expected,
            "original_episodes": str(directory / "evaluation/episodes.jsonl"),
            "original_episodes_sha256": hashlib.sha256((directory / "evaluation/episodes.jsonl").read_bytes()).hexdigest()}
    root = cli.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    for name, digest in source_manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    shutil.copyfile(Path(__file__), root / "runner.py")
    write_json(root / "manifest.json", {"training_seeds": seeds, "episodes_per_seed": episodes,
        "checkpoints": checkpoints, "smoke_only": cli.smoke, "source_study": str(SOURCE),
        "source_sha256": source_manifest["source_sha256"], "agents": 3, "landmarks": 3,
        "loss": 0., "min_agent_distance": .4, "min_landmark_distance": .4, "horizon": 100,
        "communication_protocol": "position-velocity-all-to-all-v0", "terminate_on_collision": False,
        "intervention": "Force all transmit gates to zero after original deterministic joint-action selection; movement logits, weights and recurrent memory unchanged"})
    rows, errors = [], {}
    aggregate(root, rows, seeds, cli.smoke)
    total = len(seeds) * episodes * 2
    with ProcessPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(worker, str(root), seed): seed for seed in seeds}
        pending = set(futures)
        while pending:
            for future in list(pending):
                if future.done():
                    pending.remove(future)
                    try:
                        rows.append(future.result())
                    except Exception:
                        errors[str(futures[future])] = traceback.format_exc()
                    aggregate(root, rows, seeds, cli.smoke)
            progress(root, total, "running" if pending else "failed" if errors else "complete", errors)
            if pending:
                time.sleep(5)
    for name in ("episodes.jsonl", "paired-episodes.jsonl"):
        with (root / name).open("x", encoding="utf-8") as stream:
            for seed in seeds:
                if str(seed) not in errors:
                    stream.write((root / f"seed-{seed}" / name).read_text())
    if errors:
        raise SystemExit("Some ablations failed; see status.json")


if __name__ == "__main__":
    main()
