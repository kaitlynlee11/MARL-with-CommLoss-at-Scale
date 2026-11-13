"""Extend frozen baseline evaluations from 200 to 500 paired held-out scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from run_adaptive_seeds import write_json, atomic_text

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/baseline-agent-spacing04-15-seeds-20261109"
OUTPUT = REPO / "results/baseline-heldout500-20261112"
POLICIES = {"full": "full", "none": "none", "fixed_5": "fixed"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(root_string, seed, policy):
    root = Path(root_string)
    sys.path.insert(0, str(root / "source"))
    import torch
    from limited_comm.mappo import load_policy
    from limited_comm.evaluate import evaluate_episode, summarize
    torch.set_num_threads(1)
    manifest = json.loads((root / "manifest.json").read_text())
    item = manifest["checkpoints"][f"{seed}/{policy}"]
    checkpoint = Path(item["path"])
    assert sha(checkpoint) == item["sha256"]
    model, saved = load_policy(checkpoint)
    config = saved["config"]
    assert config["seed"] == seed
    assert config["min_agent_distance"] == config["min_landmark_distance"] == .4
    assert config["horizon"] == 100 and config["loss"] == 0
    assert config["training_schedule"] == POLICIES[policy]
    directory = root / f"seed-{seed}" / policy
    directory.mkdir(parents=True, exist_ok=True)
    original = [json.loads(line) for line in Path(item["episodes"]).read_text().splitlines()]
    assert len(original) == 200 and [r["seed"] for r in original] == list(range(400000, 400200))
    def evaluate(scenario):
        return {"condition":policy, **evaluate_episode(model, scenario, POLICIES[policy], period=5, loss=0,
                                horizon=100, radius=.6, min_landmark_distance=.4,
                                min_agent_distance=.4, terminate_on_collision=False)}
    # Confirm the copied historical evaluator exactly reproduces a saved scenario.
    reproduced = evaluate(400000)
    for key, value in original[0].items():
        if isinstance(value, float):
            assert abs(reproduced[key] - value) < 1e-9, (key, reproduced[key], value)
        else:
            assert reproduced[key] == value, key
    path = directory / "episodes.jsonl"
    if not path.exists():
        atomic_text(path, "".join(json.dumps(r) + "\n" for r in original))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[:200] == original
    assert [r["seed"] for r in records] == list(range(400000, 400000 + len(records)))
    assert 200 <= len(records) <= 500
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for scenario in range(400000 + len(records), 400500):
            record = evaluate(scenario)
            assert record["actual_min_separation"] >= .4
            assert record["actual_initial_agent_separation"] >= .4
            assert record["task_reward_version"] == "legacy-distance-v0"
            assert record["attempted_links"] == record["delivered_links"] == {"full":600,"none":0,"fixed_5":120}[policy]
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            if len(records) % 25 == 0:
                write_json(directory / "status.json", {"episodes":len(records), "target":500})
    assert sha(checkpoint) == item["sha256"]
    result = {"training_seed": seed, "condition": policy, **summarize(records),
              "provenance": f"seed-{seed}/{policy}", "checkpoint_sha256":item["sha256"],
              "original_200_preserved":True, "original_scenario_reproduced":True}
    write_json(directory / "summary.json", result)
    write_json(directory / "status.json", {"episodes":500, "target":500, "complete":True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.exists():
        shutil.copytree(BASE / "source", OUTPUT / "source", dirs_exist_ok=True)
        checkpoints = {}
        for seed in range(42,57):
            for policy in POLICIES:
                directory = BASE / f"seed-{seed}" / policy
                checkpoint = directory / "best.pt"
                episodes = directory / "evaluation/episodes.jsonl"
                checkpoints[f"{seed}/{policy}"] = {"path":str(checkpoint), "sha256":sha(checkpoint),
                                                     "episodes":str(episodes), "episodes_sha256":sha(episodes)}
        write_json(manifest_path, {"source_study":str(BASE), "checkpoints":checkpoints,
            "source_sha256":{p.relative_to(OUTPUT / "source").as_posix():sha(p) for p in (OUTPUT / "source").rglob("*.py")},
            "training_seeds":list(range(42,57)), "scenario_seeds":[400000,400499],
            "reused_episodes_per_policy":200, "additional_episodes_per_policy":300,
            "min_agent_distance":.4, "min_landmark_distance":.4, "loss":0,
            "training":False, "checkpoint_selection":"unchanged validation-selected best.pt"})
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest["source_sha256"].items():
        assert sha(OUTPUT / "source" / relative) == digest
    for item in manifest["checkpoints"].values():
        assert sha(Path(item["path"])) == item["sha256"]
        assert sha(Path(item["episodes"])) == item["episodes_sha256"]
    started = time.time()
    rows = []
    def progress(complete=False):
        count = 0
        for seed in range(42,57):
            for policy in POLICIES:
                path = OUTPUT / f"seed-{seed}" / policy / "status.json"
                if path.exists():
                    count += json.loads(path.read_text())["episodes"]
        write_json(OUTPUT / "status.json", {"state":"complete" if complete else "running",
            "completed_condition_runs":len(rows), "total_condition_runs":45,
            "episodes_ready":count, "total_episodes":22500, "additional_episodes":13500,
            "elapsed_seconds":time.time()-started, "updated_at":time.time()})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(worker, str(OUTPUT), seed, policy) for seed in range(42,57) for policy in POLICIES}
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
            progress()
    # Independent aggregation checks cover all 22,500 original plus added episodes.
    aggregate = []
    for row in sorted(rows, key=lambda r:(r["training_seed"],r["condition"])):
        records = [json.loads(line) for line in (OUTPUT / row["provenance"] / "episodes.jsonl").read_text().splitlines()]
        assert len(records) == 500
        assert [r["seed"] for r in records] == list(range(400000,400500))
        for record in records:
            assert record["actual_min_separation"] >= .4 and record["actual_initial_agent_separation"] >= .4
            assert record["loss"] == 0 and record["active_steps"] == 100 and not record["terminate_on_collision"]
            aggregate.append({"training_seed":row["training_seed"], "condition":row["condition"], **record})
    atomic_text(OUTPUT / "episodes.jsonl", "".join(json.dumps(r)+"\n" for r in aggregate))
    from run_baseline_seeds import report
    report(OUTPUT, rows, 15, measured_only=True, document_path=OUTPUT / "report-copy.md", evaluation_episodes=500)
    text = (OUTPUT / "report.md").read_text()
    marker = "All included rows are freshly measured from checkpoint evaluations. Seed 42 is rerun from scratch using the same source snapshot, training budgets and held-out scenarios as seeds 43-56. Historical rounded values are excluded."
    text = text.replace(marker, "Expanded held-out evaluation: 500 episodes per trained policy (22,500 total). The original 200 scenarios are preserved exactly, with 300 additional scenarios (400200–400499) evaluated using the same frozen checkpoints and historical environment code. No retraining or checkpoint reselection was performed. The original 200-episode study remains archived at `results/baseline-agent-spacing04-15-seeds-20261109/`; this expanded study is at `results/baseline-heldout500-20261112/`.")
    for path in (OUTPUT / "report.md", REPO / "docs/baseline-15-seeds-results.md", REPO / "docs/navigation-baseline-15-seeds-results.md"):
        atomic_text(path, text)
    from baseline_episode_variance import main as episode_variance
    episode_variance(OUTPUT, 500)
    (OUTPUT / "report-copy.md").unlink()
    write_json(OUTPUT / "verification.json", {"passed":True, "condition_runs":45, "episodes":22500,
        "preserved_original_episodes":9000, "new_episodes":13500, "paired_scenarios":500,
        "reproduced_original_scenarios":45, "agent_and_landmark_min_spacing":.4,
        "frozen_checkpoints_verified":45, "source_hashes_verified":True})
    progress(complete=True)
    print("Completed and verified 45 policies x 500 held-out episodes.", flush=True)


if __name__ == "__main__":
    main()
