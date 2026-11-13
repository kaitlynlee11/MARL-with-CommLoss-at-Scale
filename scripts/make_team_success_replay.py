"""Replay the first saved successful adaptive N=20 case, without reselection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scheduler-team-sizes-zero-loss-20261111"


def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    successful = []
    for seed in manifest["training_seeds"]:
        for line in (ROOT / f"n-20/seed-{seed}/episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["condition"] == "adaptive" and row["final_success"]:
                successful.append(row)
    record = min(successful, key=lambda r: (r["training_seed"], r["seed"]))
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(ROOT / "source"))
    from limited_comm.layout import minimum_separation
    from limited_comm.mappo import load_policy
    # Reuse the enhanced frame collector with the exact frozen environment and
    # baseline reward implementation. Never modify the evaluation snapshot.
    spec = importlib.util.spec_from_file_location("limited_comm.team_replay_collector", REPO / "src/limited_comm/visualize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = manifest["checkpoints"][str(record["training_seed"])]["adaptive"]
    checkpoint = Path(provenance["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == provenance["sha256"]
    model, _ = load_policy(checkpoint)
    replay = module.collect_replay(20, 0., 100, 5, record["seed"], model=model,
                                  min_agent_distance=.4, min_landmark_distance=.4, layout_sampler="sequential")
    run = replay["runs"][0]
    first, final = run["frames"][0], run["frames"][-1]
    coords = np.asarray([*first["agents"], *first["landmarks"]], dtype="<f8")
    assert hashlib.sha256(coords.tobytes()).hexdigest() == record["initial_layout_sha256"]
    assert run["active_steps"] == record["active_steps"] == 100
    assert final["success"] and final["coverage"] == record["final_coverage"] == 1
    assert final["attempts"] == record["attempted_links"]
    assert final["delivered"] == record["delivered_links"]
    assert final["total_collisions"] == record["collision_pairs"]
    assert sum(f["collision_pairs"] > 0 for f in run["frames"][1:]) / 100 == record["collision_step_rate"]
    assert abs(final["task_return"] - record["task_return"]) < 1e-8
    assert next(i for i, f in enumerate(run["frames"][1:], 1) if f["success"]) == record["first_success_step"]
    assert minimum_separation(coords[:20]) >= .4 and minimum_separation(coords[20:]) >= .4
    assert all(f["landmarks"] == first["landmarks"] for f in run["frames"])
    replay.update(training_seed=record["training_seed"], checkpoint_sha256=provenance["sha256"],
                  logged_episode=record, verified=True,
                  selection_note="First saved final-success adaptive N=20 episode by training seed then scenario seed; selected for illustration, not an aggregate success claim.")
    filename = f"adaptive-20-success-policy{record['training_seed']}-scenario{record['seed']}"
    output = ROOT / "replays"
    output.mkdir(exist_ok=True)
    (output / f"{filename}.json").write_text(json.dumps(replay, allow_nan=False), encoding="utf-8")
    run.update(label="Successful 20-agent adaptive episode",
               description=f"Training seed {record['training_seed']}; scenario {record['seed']}. Selected saved success; collisions are allowed under the evaluation rule.")
    standalone = module.HTML.replace("__REPLAY_DATA__", json.dumps(replay, allow_nan=False).replace("<", "\\u003c"))
    standalone = standalone.replace("<h1>Watch communication schedules</h1>", "<h1>Successful 20-agent adaptive replay</h1>")
    (output / f"{filename}.html").write_text(standalone, encoding="utf-8")
    print(json.dumps({"replay_json": str(output / f"{filename}.json"), "training_seed": record["training_seed"],
                      "scenario": record["seed"], "first_success_step": record["first_success_step"],
                      "final_coverage": final["coverage"], "collision_step_rate": record["collision_step_rate"],
                      "packets": final["attempts"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
