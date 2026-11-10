"""Progress is derived from live logs, with verification kept separate."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "serve_progress", Path(__file__).resolve().parents[1] / "scripts/serve_progress.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_uncreated_condition_is_queued(tmp_path):
    row = progress.condition_progress(tmp_path / "absent", 42, "full")
    assert row["state"] == "queued" and row["steps"] == 0


def test_incomplete_log_line_does_not_lose_last_completed_update(tmp_path):
    log = tmp_path / "initial/learning.jsonl"
    save(log, {"steps": 6400, "validation": {"success": .5}})
    with log.open("a") as stream:
        stream.write('{"steps": 128')
    row = progress.condition_progress(tmp_path, 42, "full")
    assert row["state"] == "initial" and row["steps"] == 6400
    assert row["validation"]["success"] == .5


def test_refinement_steps_are_cumulative_not_double_counted(tmp_path):
    save(tmp_path / "initial/learning.jsonl", {"steps": 8320000})
    save(tmp_path / "refine/learning.jsonl", {"steps": 8384000})
    row = progress.condition_progress(tmp_path, 42, "full")
    assert row["state"] == "refine"
    assert row["steps"] == 8384000 and row["refine_steps"] == 64000


def test_evaluation_is_separate_from_verified_completion(tmp_path):
    save(tmp_path / "refine/learning.jsonl", {"steps": progress.TOTAL})
    save(tmp_path / "evaluation/episodes.jsonl", {"final_success": True})
    assert progress.condition_progress(tmp_path, 42, "full")["state"] == "evaluating"
    save(tmp_path / "evaluation/summary.json", {"conditions": {"full": {"success_rate": .9}}})
    assert progress.condition_progress(tmp_path, 42, "full")["state"] == "verifying"
    row = progress.condition_progress(tmp_path, 42, "full", verified=True)
    assert row["state"] == "complete" and row["episodes"] == 1
    assert row["fraction"] == 1


def test_collector_uses_fresh_seed42_and_excludes_historical_aggregate(tmp_path):
    original = tmp_path / "results/baseline-15-seeds-20261109"
    measured = tmp_path / "results/baseline-agent-spacing04-15-seeds-20261109"
    save(original / "seed-42/full/initial/learning.jsonl", {"steps": 999999})
    save(measured / "seed-42/full/initial/learning.jsonl", {"steps": 6400})
    save(original / "summary.json", {"per_seed": [{"training_seed": 42, "condition": "full"}]})
    data = progress.collect_progress(tmp_path)
    assert data["total_runs"] == 45 and data["training_budget"] == 403200000
    seed42 = next(r for r in data["runs"] if r["seed"] == 42 and r["condition"] == "full")
    assert seed42["steps"] == 6400 and seed42["state"] == "initial"
    assert data["counts"]["complete"] == 0
