"""Independently audit a completed frozen-policy packet-loss comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_adaptive_seeds import write_json


def expected_deliveries(seed, loss, method):
    draws = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]).random((100, 3, 3))
    mask = ~np.eye(3, dtype=bool)
    steps = np.ones(100, dtype=bool) if method == "full" else np.arange(100) % 5 == 0
    return int(((draws >= loss) & mask & steps[:, None, None]).sum())


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"]
    sys.path.insert(0, str(root / "source"))
    from evaluate_scheduler_losses import METRICS, summarize_rows
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    for sources in manifest["checkpoints"].values():
        for info in sources.values():
            assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
    records = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    expected = len(manifest["training_seeds"]) * 9 * manifest["episodes_per_condition"]
    assert len(records) == expected
    keys = [(r["training_seed"], r["loss"], r["condition"], r["seed"]) for r in records]
    assert len(set(keys)) == len(keys)
    per_seed = {(r["training_seed"], r["loss"], r["condition"]): r for r in summary["per_seed"]}
    packet_checks = 0
    layouts = {}
    groups = {}
    for row in records:
        key = (row["training_seed"], row["loss"], row["condition"])
        groups.setdefault(key, []).append(row)
        assert row["actual_initial_agent_separation"] >= .4 and row["actual_min_separation"] >= .4
        assert not row["initial_collision"] and row["active_steps"] == 100 and not row["terminate_on_collision"]
        assert row["final_success"] == (row["final_coverage"] == 1)
        assert row["any_collision"] == (row["collision_pairs"] > 0)
        assert row["payload_bytes"] == 16 * row["attempted_links"]
        assert 0 <= row["delivered_links"] <= row["attempted_links"] <= 600
        layout = (row["actual_initial_agent_separation"], row["actual_min_separation"])
        assert layouts.setdefault(row["seed"], layout) == layout
        if row["condition"] != "adaptive":
            assert row["delivered_links"] == expected_deliveries(row["seed"], row["loss"], row["condition"])
            packet_checks += 1
    assert len(groups) == len(manifest["training_seeds"]) * 9
    for key, rows in groups.items():
        assert [r["seed"] for r in rows] == list(range(400000, 400000 + manifest["episodes_per_condition"]))
        actual = summarize_rows(rows)
        assert all(np.isclose(actual[m], per_seed[key][m]) for m in METRICS)
    for condition in summary["conditions"]:
        subset = [r for key, r in per_seed.items() if key[1:] == (condition["loss"], condition["condition"])]
        assert len(subset) == len(manifest["training_seeds"])
        for metric in METRICS:
            assert np.isclose(np.mean([r[metric] for r in subset]), condition["metrics"][metric]["mean"])
    result = {"verified_unix": time.time(), "episodes": len(records), "independent_fixed_schedule_delivery_checks": packet_checks,
              "source_and_checkpoint_hashes_unchanged": True, "paired_layouts": True,
              "summary_recomputed": True, "all_checks_passed": True}
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root.resolve())
