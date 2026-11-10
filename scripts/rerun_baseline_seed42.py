"""Rerun seed 42, and aggregate only verified measurements for seeds 42-56.

The existing 43-56 study continues unchanged. This coordinator has its own
report/output, preventing the original runner from restoring rounded seed-42
rows. It exits only when all 45 condition runs are verified, or a job fails.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from run_baseline_seeds import LABELS, REPO, report, verify_condition, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, default=REPO / "results/baseline-15-seeds-20261109")
    parser.add_argument("--output", type=Path, default=REPO / "results/navigation-baseline-15-seeds-20261109")
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    source = cli.source_study.resolve()
    root = cli.output.resolve()
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original["smoke_only"] != cli.smoke:
        parser.error("Smoke and production source studies cannot be mixed")
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "source"
    shutil.copytree(source / "source", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    for name, digest in original["source_sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source snapshot hash mismatch: {name}")
    shutil.copyfile(Path(__file__), root / "runner.py")
    shutil.copyfile(REPO / "scripts/run_baseline_seeds.py", root / "analysis.py")
    seeds = [42, *original["new_training_seeds"]]
    write_json(root / "manifest.json", {
        "training_seeds": seeds, "rerun_training_seed": 42,
        "other_seeds_source_study": str(source), "smoke_only": cli.smoke,
        "source_sha256": original["source_sha256"], "historical_rounded_results_included": False,
        "initial_steps": 6400 if cli.smoke else 8320000,
        "refine_steps": 6400 if cli.smoke else 640000,
        "evaluation_episodes_per_policy": 2 if cli.smoke else 200,
        "protocol": "Exactly the source snapshot and settings used for the other 14 seeds; legacy distance reward, agent spacing 0, landmark spacing 0.4, loss 0, 100-step horizon."})
    sys.path.insert(0, str(snapshot))
    os.environ["PYTHONPATH"] = str(snapshot)
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from limited_comm.study import train_condition
    torch.set_num_threads(1)
    args = SimpleNamespace(root=root, initial_steps=6400 if cli.smoke else 8320000,
                           refine_steps=6400 if cli.smoke else 640000,
                           episodes=2 if cli.smoke else 200, curriculum_steps=1000000,
                           min_landmark_distance=.4, min_agent_distance=0., test_seed=400000,
                           training_seed=42)
    (root / "seed-42").mkdir()
    document = REPO / "docs/navigation-baseline-15-seeds-results.md"
    rows, verified, failed = [], set(), {}
    started = time.time()

    def refresh():
        for seed in seeds:
            study_root = root if seed == 42 else source
            for label in LABELS:
                key = (seed, label)
                directory = study_root / f"seed-{seed}" / label
                if key in verified or not (directory / "evaluation/summary.json").exists():
                    continue
                settings = SimpleNamespace(**{**vars(args), "root": study_root})
                row = verify_condition(directory, seed, label, settings)
                row["provenance"] = str(directory)
                rows.append(row)
                verified.add(key)
        report(root, rows, len(seeds), smoke=cli.smoke, measured_only=True, document_path=document)
        write_json(root / "status.json", {
            "state": "failed" if failed else "complete" if len(verified) == len(seeds) * 3 else "running",
            "started_unix": started, "updated_unix": time.time(),
            "verified_condition_runs": len(verified), "expected_condition_runs": len(seeds) * 3,
            "seed42": {label: "complete" if (42, label) in verified else failed.get(label, "running") for label in LABELS},
            "failures": failed, "source_study": str(source)})

    refresh()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(train_condition, label, schedule, root / "seed-42", args): label
                   for label, schedule in LABELS.items()}
        collected = set()
        while True:
            for future, label in futures.items():
                if future.done() and future not in collected:
                    collected.add(future)
                    try:
                        future.result()
                    except Exception as error:
                        failed[label] = repr(error)
            source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
            source_failures = {key: value for key, value in source_status["jobs"].items() if value.startswith("failed")}
            failed.update(source_failures)
            refresh()
            if failed or len(verified) == len(seeds) * 3:
                break
            time.sleep(20)
    if failed:
        raise SystemExit("A baseline job failed; see status.json and condition logs")
    # All records are measured. Preserve the training seed on every episode.
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: (r["training_seed"], r["condition"])):
            directory = Path(row["provenance"])
            for line in (directory / "evaluation/episodes.jsonl").read_text(encoding="utf-8").splitlines():
                stream.write(json.dumps({"training_seed": row["training_seed"], **json.loads(line)}) + "\n")
    refresh()
    print(f"Completed {len(seeds)} measured training seeds per baseline: {root}", flush=True)


if __name__ == "__main__":
    main()
