"""Test frozen spaced-navigation checkpoints under independent packet loss."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate_episode, summarize
from .layout import minimum_separation
from .mappo import load_policy
from .visualize import HTML, collect_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/spaced-navigation"))
    parser.add_argument("--output", type=Path, default=Path("results/spaced-navigation-loss25"))
    parser.add_argument("--loss", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.loss <= 1:
        parser.error("loss must be in [0, 1]")
    baseline = json.loads((args.source / "summary.json").read_text(encoding="utf-8"))
    old_rows = [json.loads(line) for line in
                (args.source / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    if baseline["loss"] != 0 or baseline["min_landmark_distance"] != 0.4:
        parser.error("source must be the loss-free, 0.4-spaced baseline study")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    summaries, provenance, paired, records, models = {}, {}, {}, [], {}
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for label, schedule in [("full", "full"), ("none", "none"), ("fixed_5", "fixed")]:
            path = args.source / label / "best.pt"
            model, checkpoint = load_policy(path)
            config = checkpoint["config"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == baseline["selections"][label]["sha256"]
            assert config["min_landmark_distance"] == 0.4 and config["radius"] == 0.6
            assert config["training_schedule"] == schedule
            assert config["horizon"] == 100 and config.get("period", 5) == 5
            models[schedule] = model
            provenance[label] = {"checkpoint": str(path.resolve()), "sha256": digest,
                                 "steps": checkpoint["steps"], "training_config": config}
            originals = {r["seed"]: r for r in old_rows if r["condition"] == label}
            assert len(originals) == baseline["episodes_per_condition"]
            rows = []
            for i, (seed, original) in enumerate(sorted(originals.items())):
                row = {"condition": label, **evaluate_episode(
                    model, seed, schedule, 5, args.loss, 100, 0.6, min_landmark_distance=0.4)}
                assert row["actual_min_separation"] == original["actual_min_separation"]
                assert row["attempted_links"] == original["attempted_links"]
                if schedule == "none":
                    assert all(row[k] == original[k] for k in original if k != "loss")
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"{label}: {i + 1}/{len(originals)} episodes", flush=True)
            records.extend(rows)
            summaries[label] = summarize(rows)
            attempts = sum(r["attempted_links"] for r in rows)
            delivered = sum(r["delivered_links"] for r in rows)
            summaries[label]["observed_packet_loss"] = 1 - delivered / attempts if attempts else None
            paired[label] = {
                "success_to_failure": sum(originals[r["seed"]]["final_success"] and not r["final_success"] for r in rows),
                "failure_to_success": sum(not originals[r["seed"]]["final_success"] and r["final_success"] for r in rows),
                "success_change_pp": 100 * (summaries[label]["success_rate"] - baseline["conditions"][label]["success_rate"])}
            print(label, json.dumps(summaries[label]), flush=True)
    result = {"source": str(args.source.resolve()), "loss": args.loss,
              "min_landmark_distance": 0.4, "radius": 0.6, "horizon": 100,
              "episodes_per_condition": baseline["episodes_per_condition"],
              "test_seed_start": baseline["test_seed_start"],
              "scope": "frozen loss-free-trained policies; no retraining or checkpoint selection",
              "checkpoints": provenance, "conditions": summaries, "paired_changes": paired,
              "loss_free_conditions": baseline["conditions"]}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    seeds = {"comparison.html": baseline["test_seed_start"]}
    # Also show the first held-out case where loss changes a communication policy's success.
    affected = [r["seed"] for r in records if r["condition"] != "none" and
                next(o["final_success"] for o in old_rows if o["seed"] == r["seed"] and
                     o["condition"] == r["condition"]) and not r["final_success"]]
    if affected:
        seeds["loss-affected.html"] = min(affected)
    for filename, seed in seeds.items():
        replay = collect_replay(3, args.loss, 100, 5, seed, models=models, min_landmark_distance=0.4)
        for run in replay["runs"]:
            row = next(r for r in records if r["seed"] == seed and r["schedule"] == run["schedule"])
            final = run["frames"][-1]
            assert final["success"] == row["final_success"]
            assert abs(final["task_return"] - row["task_return"]) < 1e-6
            assert final["delivered"] == row["delivered_links"]
            landmarks = np.asarray([f["landmarks"] for f in run["frames"]])
            assert (minimum_separation(landmarks) >= 0.4).all()
            assert (landmarks == landmarks[0]).all()
        data = json.dumps(replay, allow_nan=False).replace("<", "\\u003c")
        (args.output / filename).write_text(HTML.replace("__REPLAY_DATA__", data), encoding="utf-8")
    lines = ["# Packet-loss evaluation of spaced-landmark MAPPO policies", "",
             f"Frozen checkpoints from the loss-free study, tested with {args.loss:.0%} independent packet loss per directed link per sending step.",
             "Each separately trained policy keeps its own schedule; no additional training or checkpoint selection was performed.",
             f"Same {baseline['episodes_per_condition']} held-out seeds per condition, starting at {baseline['test_seed_start']}; three agents, 100 steps.",
             "Nearby-only landmark sensing (radius 0.6); static landmarks with pairwise Euclidean center distance >= 0.4.", "",
             "| Schedule | Success, no loss | Success, with loss (95% interval) | Change (pp) | Coverage | Task return | Attempted / delivered packets | Collision step rate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, summary in summaries.items():
        lo, hi = summary["success_95pct_wilson"]
        lines.append(f"| {label} | {baseline['conditions'][label]['success_rate']:.1%} | "
                     f"{summary['success_rate']:.1%} ({lo:.1%} to {hi:.1%}) | "
                     f"{paired[label]['success_change_pp']:+.1f} | {summary['final_coverage']:.1%} | "
                     f"{summary['task_return']:.2f} | {summary['attempted_links']:.0f} / "
                     f"{summary['delivered_links']:.2f} | {summary['collision_step_rate']:.1%} |")
    lines.extend(["", "Final success means all three landmarks are covered simultaneously at step 100 (coverage radius 0.1).",
                  f"Minimum observed landmark separation: {min(r['actual_min_separation'] for r in records):.6f}.",
                  "Failed deliveries retain older cached data until its five-step expiration; no retransmission is added.",
                  "Packets carry 16 payload bytes; attempted bytes still count when delivery fails. Task return excludes communication penalties.",
                  "The no-communication trajectories and metrics match the loss-free baseline exactly.",
                  "One training seed per schedule. Wilson intervals describe episode uncertainty, not variation across training seeds.",
                  "This measures test-time robustness to loss, rather than performance after training with loss.", "",
                  "## Paired episode changes", "",
                  "| Schedule | Success becomes failure | Failure becomes success |",
                  "| --- | --- | --- |"])
    for label, changes in paired.items():
        lines.append(f"| {label} | {changes['success_to_failure']} | {changes['failure_to_success']} |")
    lines.extend(["", "## Delivery checks", ""])
    for label, summary in summaries.items():
        observed = summary["observed_packet_loss"]
        lines.append(f"- {label}: observed packet loss {observed:.2%}." if observed is not None else
                     f"- {label}: no attempted packets; observed packet loss is undefined.")
    lines.extend(["", "## Reproduce", "", "```powershell",
                  "uv run --extra train python -m limited_comm.loss_study --output results/spaced-navigation-loss25-repeat",
                  "```", "", "Source checkpoints: `results/spaced-navigation/{full,none,fixed_5}/best.pt`.",
                  "Results: `results/spaced-navigation-loss25/summary.json` and `episodes.jsonl` (600 records).",
                  "`comparison.html` shows the first held-out seed, selected before observing outcomes.",
                  "`loss-affected.html`, if present, deliberately selects the first seed where a previously successful communication policy fails with loss.",
                  "Checkpoint hashes/configurations are recorded in summary.json; original artifacts are preserved.", ""])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Complete: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
