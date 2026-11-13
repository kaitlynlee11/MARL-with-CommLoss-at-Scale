"""Summarize within-seed episode variation from the completed baseline study."""
import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, variance

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/baseline-agent-spacing04-15-seeds-20261109"
HEADING = "## Within-seed evaluation episode variation"


def main(root=ROOT, episode_count=200):
    root = Path(root)
    groups = defaultdict(list)
    for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        groups[episode["training_seed"], episode["condition"]].append(episode)
    assert set(groups) == {(s, p) for s in range(42, 57) for p in ("full", "none", "fixed_5")}
    rows = []
    for seed in range(42, 57):
        for policy in ("full", "none", "fixed_5"):
            episodes = groups[seed, policy]
            assert len(episodes) == episode_count
            assert {e["seed"] for e in episodes} == set(range(400000, 400000 + episode_count))
            returns = [e["task_return"] for e in episodes]
            coverage = [e["final_coverage"] for e in episodes]
            assert all(0 <= value <= 1 for value in coverage)
            n = len(episodes)
            rows.append(dict(training_seed=seed, condition=policy, episodes=n,
                             task_return_mean=mean(returns), task_return_variance=variance(returns),
                             task_return_sd=stdev(returns), coverage_mean=mean(coverage),
                             coverage_variance=variance(coverage), coverage_sd=stdev(coverage)))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "within-seed-episode-variation.csv").write_text(output.getvalue(), encoding="utf-8")
    lines = [HEADING, "",
             f"Each row uses the {episode_count} held-out evaluation episodes (scenario seeds 400000–{400000 + episode_count - 1}) for one trained policy. Variance is sample variance with denominator {episode_count - 1}; SD is its square root. These statistics describe episode variation within a seed, separately from the across-training-seed SDs above. Coverage is the final fraction of landmarks covered in each episode. Coverage SD is shown in percentage points; coverage variance uses fractions on the 0–1 scale (squared fraction units). Task-return variance is in squared return units.", "",
             "| Seed | Policy | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['training_seed']} | {r['condition']} | {r['task_return_variance']:.3f} | {r['task_return_sd']:.3f} | {r['coverage_variance']:.6f} | {100*r['coverage_sd']:.2f} |")
    relative_root = root.relative_to(REPO).as_posix()
    lines += ["", f"Full-precision episode statistics: `{relative_root}/within-seed-episode-variation.csv`. Reproduce with `.venv/Scripts/python.exe scripts/baseline_episode_variance.py --root {relative_root} --episodes {episode_count}`.", ""]
    for path in (REPO / "docs/baseline-15-seeds-results.md",
                 REPO / "docs/navigation-baseline-15-seeds-results.md", root / "report.md"):
        original = path.read_text(encoding="utf-8").split(HEADING)[0].rstrip()
        path.write_text(original + "\n\n" + "\n".join(lines), encoding="utf-8")
    print(f"Validated {len(rows)} groups / {sum(r['episodes'] for r in rows)} episodes.")
    for policy in ("full", "none", "fixed_5"):
        subset = [r for r in rows if r['condition'] == policy]
        print(policy, "return SD range", min(r['task_return_sd'] for r in subset),
              max(r['task_return_sd'] for r in subset), "coverage SD pp range",
              100 * min(r['coverage_sd'] for r in subset), 100 * max(r['coverage_sd'] for r in subset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    main(args.root.resolve(), args.episodes)
