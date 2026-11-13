"""Audit completed larger-team evaluation records and recompute summaries."""
import argparse
from collections import defaultdict
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
    assert summary["complete"], "The larger-team evaluations are not complete"
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest
    for checkpoints in manifest["checkpoints"].values():
        for checkpoint in checkpoints.values():
            assert hashlib.sha256(Path(checkpoint["path"]).read_bytes()).hexdigest() == checkpoint["sha256"]
    layouts = read_json(root / "layouts.json")
    assert len(layouts) == 3 * manifest["episodes_per_condition"]
    for key, layout in layouts.items():
        n = int(key.split("/")[0])
        coords = np.asarray([*layout["agents"], *layout["landmarks"]], dtype="<f8")
        assert coords.shape == (2*n, 2) and (np.abs(coords) <= 1).all()
        assert hashlib.sha256(coords.tobytes()).hexdigest() == layout["sha256"]
        for positions in (coords[:n], coords[n:]):
            distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            assert distances.min() >= .4
    groups = defaultdict(list)
    seen, count = set(), 0
    with (root / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            n = row["agents"]
            key = (n, row["training_seed"], row["condition"])
            unique = (*key, row["seed"])
            assert unique not in seen
            seen.add(unique)
            groups[key].append(row)
            assert row["initial_layout_sha256"] == layouts[f"{n}/{row['seed']}"]["sha256"]
            assert row["landmarks"] == n and row["loss"] == 0 and row["layout_sampler"] == "sequential"
            assert row["actual_initial_agent_separation"] >= .4 and row["actual_min_separation"] >= .4
            assert not row["initial_collision"] and row["active_steps"] == 100 and not row["terminate_on_collision"]
            assert row["final_success"] == (row["final_coverage"] == 1)
            assert row["any_collision"] == (row["collision_pairs"] > 0)
            assert row["attempted_links"] == row["delivered_links"]
            assert row["payload_bytes"] == 16 * row["attempted_links"]
            denominator = n * (n-1) * 100
            assert row["packet_ratio_fixed_denominator"] == row["attempted_links"] / denominator
            if row["condition"] != "adaptive":
                assert row["attempted_links"] == {"full": denominator, "fixed_5": denominator//5, "none": 0}[row["condition"]]
            count += 1
    assert count == len(manifest["training_seeds"]) * 12 * manifest["episodes_per_condition"]
    sys.path.insert(0, str(root / "source"))
    from limited_comm.evaluate import summarize
    for saved in summary["per_seed"]:
        rows = groups[saved["agents"], saved["training_seed"], saved["condition"]]
        assert [r["seed"] for r in rows] == list(range(400000, 400000+manifest["episodes_per_condition"]))
        computed = summarize(rows)
        computed.update(task_return_per_agent=computed["task_return"] / saved["agents"],
            collision_episode_fraction=float(np.mean([r["any_collision"] for r in rows])),
            collision_pairs_per_step=float(np.mean([r["collision_pairs"] / 100 for r in rows])),
            packet_ratio_fixed_denominator=float(np.mean([r["packet_ratio_fixed_denominator"] for r in rows])))
        for metric, value in computed.items():
            assert np.allclose(value, saved[metric]), metric
    for condition in summary["conditions"]:
        subset = [r for r in summary["per_seed"] if r["agents"] == condition["agents"] and r["condition"] == condition["condition"]]
        assert len(subset) == len(manifest["training_seeds"])
        for metric, stats in condition["metrics"].items():
            assert np.isclose(stats["mean"], np.mean([r[metric] for r in subset]))
    write_json(root / "verification.json", {"verified_unix": time.time(), "episodes": count,
               "layouts": len(layouts), "checkpoint_and_source_hashes_unchanged": True,
               "paired_layouts_and_both_spacing_rules_verified": True, "packet_accounting_verified": True,
               "summaries_recomputed": True, "all_checks_passed": True})
    print(f"Verified {count:,} episodes and {len(layouts)} strictly spaced paired layouts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait", action="store_true", help="Wait for an already running study; never launches evaluations")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait:
        while True:
            status = read_json(root / "status.json", {})
            if status.get("state") == "failed":
                raise SystemExit("Evaluation failed; see status.json")
            if status.get("state") == "complete" and (root / "episodes.jsonl").exists():
                with (root / "episodes.jsonl").open("rb") as stream:
                    if sum(1 for line in stream if line.endswith(b"\n")) == status["total_episodes"]:
                        break
            time.sleep(15)
    verify(root)
