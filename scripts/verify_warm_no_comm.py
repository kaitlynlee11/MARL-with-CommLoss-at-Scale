"""Independently verify completed matched-budget no-communication controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_adaptive_seeds import read_json, write_json


def verify(root):
    manifest = read_json(root / "manifest.json")
    summary = read_json(root / "summary.json")
    assert summary["complete"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
        if name != "limited_comm/train.py":
            assert digest == manifest["original_source_sha256"][name]
    sys.path.insert(0, str(root / "source"))
    import torch
    from limited_comm.mappo import load_policy
    from train_warm_no_comm import check_initialization, summarize_rows, METRICS
    torch.set_num_threads(1)
    count, optimizer_total = 0, 0
    for result in summary["per_seed"]:
        seed = result["training_seed"]
        plan = manifest["plans"][str(seed)]
        directory = root / f"seed-{seed}"
        check_initialization(directory, plan)
        model, checkpoint = load_policy(directory / "training/last.pt")
        assert not model.spec["adaptive"] and model.action_dim == 5
        assert checkpoint["steps"] == result["control_world_steps"]
        assert checkpoint["config"]["no_early_stop"] and checkpoint["config"]["training_schedule"] == "none"
        if not manifest["smoke_only"]:
            assert checkpoint["steps"] == plan["additional_world_steps"]
            assert result["optimizer_steps"] == plan["optimizer_steps"]
        assert all(int(s["step"].item()) == result["optimizer_steps"] for s in checkpoint["optimizer"]["state"].values())
        assert hashlib.sha256((directory / "training/last.pt").read_bytes()).hexdigest() == result["checkpoint_sha256"]
        logs = [json.loads(line) for line in (directory / "training/learning.jsonl").read_text().splitlines()]
        assert len(logs) == result["ppo_updates"] and all(r["training_attempted_links"] == 0 for r in logs)
        records = [json.loads(line) for line in (directory / "evaluation/episodes.jsonl").read_text().splitlines()]
        assert len(records) == (2 if manifest["smoke_only"] else 200)
        assert [r["seed"] for r in records] == list(range(400000, 400000 + len(records)))
        assert all(r["attempted_links"] == r["delivered_links"] == 0 and not r["initial_collision"]
                   and r["actual_initial_agent_separation"] >= .4 and r["actual_min_separation"] >= .4
                   and r["active_steps"] == 100 and r["task_reward_version"] == "legacy-distance-v0" for r in records)
        computed = summarize_rows(records)
        assert all(np.isclose(computed[m], result["groups"]["warm_none"][m]) for m in METRICS)
        for name, reference in plan.get("references", {}).items():
            path = Path(reference["episodes_path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["episodes_sha256"]
            assert hashlib.sha256(Path(reference["checkpoint_path"]).read_bytes()).hexdigest() == reference["checkpoint_sha256"]
            original = [json.loads(line) for line in path.read_text().splitlines()][:len(records)]
            for a, b in zip(original, records):
                assert a["seed"] == b["seed"]
                assert a["actual_initial_agent_separation"] == b["actual_initial_agent_separation"]
                assert a["actual_min_separation"] == b["actual_min_separation"]
            metrics = summarize_rows(original)
            assert all(np.isclose(metrics[m], result["groups"][name][m]) for m in METRICS)
        count += len(records)
        optimizer_total += result["optimizer_steps"]
    for name, metrics in summary["groups"].items():
        for metric, stat in metrics.items():
            assert np.isclose(stat["mean"], np.mean([r["groups"][name][metric] for r in summary["per_seed"]]))
    for name, metrics in summary["paired_differences"].items():
        for metric, stat in metrics.items():
            assert np.isclose(stat["mean"], np.mean([r["differences"][name][metric] for r in summary["per_seed"]]))
    write_json(root / "verification.json", {"verified_unix": time.time(), "seeds": len(summary["per_seed"]),
        "control_episodes": count, "optimizer_steps": optimizer_total,
        "initializer_actor_and_adaptive_movement_marginals_match": True,
        "fresh_critics_match_adaptive_initialization": True, "optimizer_budgets_matched": True,
        "zero_training_and_evaluation_packets": True, "reference_data_unchanged": True,
        "summaries_recomputed": True, "all_checks_passed": True})
    print(f"Verified {len(summary['per_seed'])} matched controls, {count} held-out episodes and {optimizer_total} optimizer steps", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait:
        while True:
            status = read_json(root / "status.json", {})
            if status.get("state") == "failed":
                raise SystemExit("Control training failed; inspect failure.json")
            if status.get("state") == "complete" and (root / "episodes.jsonl").exists():
                manifest = read_json(root / "manifest.json")
                expected = len(manifest["training_seeds"]) * (2 if manifest["smoke_only"] else 200)
                with (root / "episodes.jsonl").open("rb") as stream:
                    if sum(1 for line in stream if line.endswith(b"\n")) == expected:
                        break
            time.sleep(15)
    verify(root)
