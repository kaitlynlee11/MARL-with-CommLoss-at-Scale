"""Audit saved control reproduction, silence, and paired ablation statistics."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from evaluate_adaptive_ablation import METRICS, check_original
from run_adaptive_seeds import read_json, write_json


def verify(root):
    manifest = read_json(root / "manifest.json")
    summary = read_json(root / "summary.json")
    assert summary["complete"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    groups, saved = defaultdict(list), {}
    for seed in manifest["training_seeds"]:
        info = manifest["checkpoints"][str(seed)]
        assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
        assert hashlib.sha256(Path(info["original_episodes"]).read_bytes()).hexdigest() == info["original_episodes_sha256"]
        saved[seed] = {r["seed"]: r for r in (json.loads(line) for line in Path(info["original_episodes"]).read_text().splitlines())}
    keys, count = set(), 0
    with (root / "episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            unique = row["training_seed"], row["seed"], row["condition"]
            assert unique not in keys
            keys.add(unique)
            groups[row["training_seed"], row["condition"]].append(row)
            assert row["active_steps"] == 100 and not row["initial_collision"]
            assert row["loss"] == 0 and row["schedule"] == "adaptive"
            assert row["task_reward_version"] == "legacy-distance-v0"
            if row["condition"] == "original":
                # Stored evaluation label differs from source metadata only.
                comparable = {**row, "condition": "adaptive"}
                check_original(comparable, saved[row["training_seed"]][row["seed"]])
            else:
                assert row["condition"] == "communication_disabled"
                assert row["attempted_links"] == row["delivered_links"] == row["payload_bytes"] == 0
            count += 1
    expected = len(manifest["training_seeds"]) * manifest["episodes_per_seed"]
    assert count == 2 * expected
    sys.path.insert(0, str(root / "source"))
    from evaluate_adaptive_ablation import summarize_rows
    per_seed = {r["training_seed"]: r for r in summary["per_seed"]}
    for seed in manifest["training_seeds"]:
        for name in ("original", "communication_disabled"):
            rows = groups[seed, name]
            assert [r["seed"] for r in rows] == list(range(400000, 400000 + manifest["episodes_per_seed"]))
            actual = summarize_rows(rows)
            assert all(np.isclose(actual[m], per_seed[seed][name][m]) for m in METRICS)
        for a, b in zip(groups[seed, "original"], groups[seed, "communication_disabled"]):
            assert a["seed"] == b["seed"]
            for key in ("actual_initial_agent_separation", "actual_min_separation"):
                assert a[key] == b[key] >= .4
            if a["attempted_links"] == 0:
                assert all(a[k] == b[k] for k in ("final_success", "collision_step_rate", "task_return", "final_coverage"))
        for m in METRICS:
            assert np.isclose(per_seed[seed]["differences_disabled_minus_original"][m],
                              per_seed[seed]["communication_disabled"][m] - per_seed[seed]["original"][m])
    for name in summary["statistics"]:
        for metric, stat in summary["statistics"][name].items():
            assert np.isclose(stat["mean"], np.mean([r[name][metric] for r in summary["per_seed"]]))
    pairs = [json.loads(line) for line in (root / "paired-episodes.jsonl").read_text().splitlines()]
    assert len(pairs) == expected
    assert sum(p["original_success"] and not p["disabled_success"] for p in pairs) == summary["success_to_failure"]
    assert sum(not p["original_success"] and p["disabled_success"] for p in pairs) == summary["failure_to_success"]
    result = {"verified_unix": time.time(), "original_rollouts_reproduced": expected,
              "communication_disabled_rollouts": expected, "paired_scenarios": expected,
              "source_and_checkpoint_hashes_unchanged": True, "disabled_packets_all_zero": True,
              "paired_statistics_recomputed": True, "all_checks_passed": True}
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root.resolve())
