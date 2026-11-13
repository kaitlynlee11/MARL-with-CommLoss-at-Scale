"""Independently audit the expanded 500-episode baseline summaries and provenance."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev, variance

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/baseline-heldout500-20261112"


def check():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    aggregate = [json.loads(line) for line in (ROOT / "episodes.jsonl").read_text().splitlines()]
    assert len(aggregate) == 22500
    summary = json.loads((ROOT / "summary.json").read_text())
    assert summary["complete"]
    with (ROOT / "within-seed-episode-variation.csv").open(newline="") as stream:
        variations = {(int(r["training_seed"]), r["condition"]):r for r in csv.DictReader(stream)}
    with (ROOT / "per-seed.csv").open(newline="") as stream:
        per_seed = {(int(r["training_seed"]), r["condition"]):r for r in csv.DictReader(stream)}
    assert len(variations) == len(per_seed) == 45
    metrics = {"success_rate":"final_success", "task_return":"task_return",
               "final_coverage":"final_coverage", "attempted_links":"attempted_links",
               "collision_step_rate":"collision_step_rate"}
    for key, item in manifest["checkpoints"].items():
        seed, policy = key.split("/")
        seed = int(seed)
        for path_key, hash_key in (("path","sha256"), ("episodes","episodes_sha256")):
            assert hashlib.sha256(Path(item[path_key]).read_bytes()).hexdigest() == item[hash_key]
        original = [json.loads(line) for line in Path(item["episodes"]).read_text().splitlines()]
        records = [r for r in aggregate if r["training_seed"] == seed and r["condition"] == policy]
        assert len(records) == 500 and [r["seed"] for r in records] == list(range(400000,400500))
        assert [{k:v for k,v in r.items() if k != "training_seed"} for r in records[:200]] == original
        for record in records:
            assert record["min_agent_distance"] == record["min_landmark_distance"] == .4
            assert record["actual_min_separation"] >= .4 and record["actual_initial_agent_separation"] >= .4
            assert record["loss"] == 0 and record["active_steps"] == 100
            assert not record["initial_collision"] and not record["terminate_on_collision"]
            assert record["attempted_links"] == record["delivered_links"] == {"full":600,"fixed_5":120,"none":0}[policy]
            assert record["task_reward_version"] == "legacy-distance-v0"
        row = per_seed[seed,policy]
        for metric, field in metrics.items():
            assert abs(float(row[metric]) - mean(r[field] for r in records)) < 1e-10
        variation = variations[seed,policy]
        for metric, field in (("task_return","task_return"), ("coverage","final_coverage")):
            values = [r[field] for r in records]
            assert abs(float(variation[metric+"_variance"]) - variance(values)) < 1e-10
            assert abs(float(variation[metric+"_sd"]) - stdev(values)) < 1e-10
    for policy, condition in summary["conditions"].items():
        for metric, stats in condition["metrics"].items():
            values = [float(r[metric]) for (_,p),r in per_seed.items() if p == policy]
            assert abs(stats["mean"] - mean(values)) < 1e-10
            assert abs(stats["sample_sd"] - stdev(values)) < 1e-10
    for relative, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / "source" / relative).read_bytes()).hexdigest() == digest
    documents = [ROOT / "report.md", REPO / "docs/baseline-15-seeds-results.md",
                 REPO / "docs/navigation-baseline-15-seeds-results.md"]
    assert len({p.read_bytes() for p in documents}) == 1
    assert "denominator 499" in documents[0].read_text(encoding="utf-8")
    print("PASS: 22,500 episodes; 9,000 originals preserved; 45 frozen checkpoints; spacing, traffic, summaries and within-seed variances verified.")


if __name__ == "__main__":
    check()
