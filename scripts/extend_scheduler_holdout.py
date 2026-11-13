"""Extend adaptive, forced-silent adaptive and warm-none evaluations to 500 scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from run_adaptive_seeds import atomic_text, read_json, write_json

REPO = Path(__file__).resolve().parents[1]
ADAPTIVE = REPO / "results/adaptive-zero-loss-15-seeds-20261110"
ABLATION = REPO / "results/adaptive-communication-ablation-20261112"
WARM = REPO / "results/warm-no-communication-15-seeds-20261112"
BASELINE = REPO / "results/baseline-heldout500-20261112"
ROOT = REPO / "results/scheduler-heldout500-20261112"
SEEDS = list(range(42,57))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(row):
    return {k:v for k,v in row.items() if k not in ("training_seed", "condition")}


def old_records(seed, group):
    if group == "adaptive":
        path = ADAPTIVE / f"seed-{seed}/evaluation/episodes.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
    elif group == "communication_disabled":
        path = ABLATION / f"seed-{seed}/episodes.jsonl"
        records = [r for line in path.read_text().splitlines()
                   if (r:=json.loads(line))["condition"] == group]
    else:
        path = WARM / f"seed-{seed}/evaluation/episodes.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 200 and [r["seed"] for r in records] == list(range(400000,400200))
    return [canonical(r) for r in records]


def worker(seed, kind):
    sys.path.insert(0, str(ROOT / "source"))
    import torch
    from limited_comm.mappo import load_policy
    from limited_comm.evaluate import evaluate_episode
    from evaluate_adaptive_ablation import ForceSilent
    torch.set_num_threads(1)
    manifest = read_json(ROOT / "manifest.json")
    info = manifest["checkpoints"][f"{seed}/{kind}"]
    assert sha(info["path"]) == info["sha256"]
    model, checkpoint = load_policy(Path(info["path"]))
    config = checkpoint["config"]
    assert config["seed"] == seed
    assert config["min_agent_distance"] == config["min_landmark_distance"] == .4
    assert config["loss"] == 0 and config["horizon"] == 100 and config["radius"] == .6
    assert not config["terminate_on_collision"] and model.spec["obs_dim"] == 28
    assert config["training_schedule"] == ("adaptive" if kind == "adaptive" else "none")
    groups = ("adaptive", "communication_disabled") if kind == "adaptive" else ("warm_none",)
    silent = ForceSilent(model) if kind == "adaptive" else None
    def evaluate(group, scenario):
        record = evaluate_episode(silent if group == "communication_disabled" else model,
            seed=scenario, schedule="adaptive" if kind == "adaptive" else "none", period=5,
            loss=0., horizon=100, radius=.6, min_agent_distance=.4, min_landmark_distance=.4,
            terminate_on_collision=False)
        assert record["actual_initial_agent_separation"] >= .4 and record["actual_min_separation"] >= .4
        assert record["active_steps"] == 100 and record["task_reward_version"] == "legacy-distance-v0"
        assert record["attempted_links"] == record["delivered_links"]
        if group != "adaptive":
            assert record["attempted_links"] == record["payload_bytes"] == 0
        return {"training_seed":seed, "condition":group, **record}
    for group in groups:
        original = old_records(seed,group)
        reproduced = canonical(evaluate(group,400000))
        for key, expected in original[0].items():
            actual = reproduced[key]
            assert abs(actual-expected) < 1e-9 if isinstance(expected,float) else actual == expected, key
        directory = ROOT / "evaluations" / group / f"seed-{seed}"
        directory.mkdir(parents=True,exist_ok=True)
        path = directory / "episodes.jsonl"
        if not path.exists():
            atomic_text(path,"".join(json.dumps({"training_seed":seed,"condition":group,**r})+"\n" for r in original))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [canonical(r) for r in records[:200]] == original
        assert len(records) <= 500 and [r["seed"] for r in records] == list(range(400000,400000+len(records)))
        with path.open("a",encoding="utf-8",buffering=1) as stream:
            for scenario in range(400000+len(records),400500):
                record = evaluate(group,scenario)
                stream.write(json.dumps(record,allow_nan=False)+"\n")
                records.append(record)
                if len(records)%25 == 0:
                    write_json(directory / "status.json",{"episodes":len(records),"target":500})
        write_json(directory / "status.json",{"episodes":500,"complete":True})
    assert sha(info["path"]) == info["sha256"]
    return {"seed":seed,"kind":kind,"groups":list(groups),"original_scenario_reproduced":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT / "manifest.json").exists():
        assert read_json(WARM / "summary.json")["complete"]
        assert read_json(BASELINE / "summary.json")["complete"]
        # Training-only edits do not change the evaluator used for warm-none.
        for p in (ADAPTIVE / "source/limited_comm").glob("*.py"):
            if p.name != "train.py":
                assert sha(p) == sha(WARM / "source/limited_comm" / p.name), p.name
        shutil.copytree(ADAPTIVE / "source",ROOT / "source",ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        checkpoints={}
        for seed in SEEDS:
            for kind, origin in (("adaptive",ADAPTIVE),("warm_none",WARM)):
                directory=origin / f"seed-{seed}"
                path=directory / (read_json(directory / "selection.json")["checkpoint"] if kind == "adaptive" else "training/last.pt")
                digest=sha(path)
                assert digest == read_json(directory / "result.json")["checkpoint_sha256"]
                checkpoints[f"{seed}/{kind}"]={"path":str(path),"sha256":digest}
        original_files={str(p):sha(p) for origin in (ADAPTIVE,WARM,ABLATION) for p in origin.glob("seed-*/**/episodes.jsonl") if "source" not in p.parts}
        original_files.update({str(p):sha(p) for p in ABLATION.glob("seed-*/episodes.jsonl")})
        write_json(ROOT / "manifest.json",{"checkpoints":checkpoints,"original_episode_sha256":original_files,
            "training_seeds":SEEDS,"scenario_seeds":[400000,400499],"episodes_per_seed":500,
            "additional_episodes_per_group_per_seed":300,"new_rollouts":13500,
            "groups":["adaptive","communication_disabled","warm_none"],"baseline_reference":str(BASELINE),
            "source_sha256":{p.relative_to(ROOT / "source").as_posix():sha(p) for p in (ROOT / "source").rglob("*.py")},
            "min_agent_distance":.4,"min_landmark_distance":.4,"loss":0,"horizon":100,
            "retraining":False,"checkpoint_reselection":False,"exploratory_landmark_protocol":False})
    manifest=read_json(ROOT / "manifest.json")
    for name,digest in manifest["source_sha256"].items():
        assert sha(ROOT / "source" / name)==digest
    for path,digest in manifest["original_episode_sha256"].items():
        assert sha(path)==digest
    results=[]
    started=time.time()
    def progress(complete=False):
        statuses=[read_json(p) for p in (ROOT / "evaluations").glob("*/seed-*/status.json")]
        write_json(ROOT / "status.json",{"state":"complete" if complete else "running", "updated_unix":time.time(),
            "completed_condition_runs":sum(bool(s.get("complete")) for s in statuses),"total_condition_runs":45,
            "episodes_ready":sum(s["episodes"] for s in statuses),"total_episodes":22500,
            "new_rollouts":13500,"elapsed_seconds":time.time()-started})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,seed,kind) for seed in SEEDS for kind in ("adaptive","warm_none")}
        while pending:
            done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            progress()
    write_json(ROOT / "evaluation-verification.json",{"completed_workers":results,"original_scenarios_reproduced":45})
    from summarize_scheduler_holdout500 import main as summarize
    summarize()
    progress(complete=True)
    print("Completed all three 500-episode held-out studies.",flush=True)


if __name__ == "__main__":
    main()
