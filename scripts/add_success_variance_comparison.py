"""Compare binary episode-success variance at 200 and 500 held-out scenarios."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from statistics import mean, variance

from run_adaptive_seeds import atomic_text

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scheduler-heldout500-20261112"
BASELINE = REPO / "results/baseline-heldout500-20261112"
HEADING = "## Success variance changes: 200 versus 500 episodes"


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def main():
    groups = defaultdict(list)
    for line in (ROOT / "episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["condition"] != "initial_none":
            groups[row["condition"], row["training_seed"]].append(row)
    for seed in range(42, 57):
        for policy in ("full", "fixed_5", "none"):
            path = BASELINE / f"seed-{seed}/{policy}/episodes.jsonl"
            groups[policy, seed] = [json.loads(line) for line in path.read_text().splitlines()]
    policies = ("adaptive", "communication_disabled", "warm_none", "full", "fixed_5", "none")
    assert len(groups) == 90
    detail = []
    summaries = []
    for policy in policies:
        subset = []
        for seed in range(42, 57):
            episodes = groups[policy, seed]
            assert len(episodes) == 500
            assert [r["seed"] for r in episodes] == list(range(400000, 400500))
            row = dict(policy=policy, training_seed=seed)
            for count in (200, 500):
                values = [int(r["final_success"]) for r in episodes[:count]]
                p = mean(values)
                v = variance(values)
                assert abs(v - count / (count - 1) * p * (1 - p)) < 1e-12
                row[f"success_rate_{count}"] = p
                row[f"success_variance_{count}"] = v
            before, after = row["success_variance_200"], row["success_variance_500"]
            row["success_variance_change_pct"] = 100 * (after - before) / before
            subset.append(row)
            detail.append(row)
        a, b = (mean(r[f"success_variance_{n}"] for r in subset) for n in (200, 500))
        c, d = (variance(r[f"success_rate_{n}"] for r in subset) for n in (200, 500))
        summaries.append(dict(policy=policy, mean_within_seed_variance_200=a,
            mean_within_seed_variance_500=b, within_seed_variance_change_pct=100*(b-a)/a,
            across_seed_variance_200=c, across_seed_variance_500=d,
            across_seed_variance_change_pct=100*(d-c)/c))
    write_csv(ROOT / "success-variance-per-seed-200-vs-500.csv", detail)
    write_csv(ROOT / "success-variance-summary.csv", summaries)
    lines = [HEADING, "",
        "Success is a binary episode outcome (1 = final-step success, 0 = failure). For each policy seed its sample variance is n/(n−1) × p × (1−p), with n equal to 200 or 500. Variance uses the 0–1 outcome scale (squared fraction units). Baseline none is identical to initial_none in the tables above and is listed once here. All six conditions use the same training seeds 42–56 and paired held-out scenarios.", "",
        "### Average within-seed success variance", "",
        "Average the 15 episode variances equally, then compute 100 × (variance500 − variance200) / variance200. This measures outcome dispersion, not uncertainty in the estimated success rate.", "",
        "| Policy | Mean success variance: 200 | Mean success variance: 500 | Change |",
        "| --- | --- | --- | --- |"]
    for r in summaries:
        lines.append(f"| {r['policy']} | {r['mean_within_seed_variance_200']:.6f} | {r['mean_within_seed_variance_500']:.6f} | {r['within_seed_variance_change_pct']:+.2f}% |")
    lines += ["", "### Variance across the 15 seed-level success rates", "",
        "This is the sample variance of the 15 per-seed success rates, rather than variance of binary episode outcomes within a seed.", "",
        "| Policy | Across-seed success variance: 200 | Across-seed success variance: 500 | Change |",
        "| --- | --- | --- | --- |"]
    for r in summaries:
        lines.append(f"| {r['policy']} | {r['across_seed_variance_200']:.8f} | {r['across_seed_variance_500']:.8f} | {r['across_seed_variance_change_pct']:+.2f}% |")
    lines += ["", "Full-precision success variance summaries: `results/scheduler-heldout500-20261112/success-variance-summary.csv`. All 90 policy/seed comparisons: `results/scheduler-heldout500-20261112/success-variance-per-seed-200-vs-500.csv`. Reproduce with `.venv/Scripts/python.exe scripts/add_success_variance_comparison.py`.", ""]
    document = REPO / "docs/scheduler-holdout-500-variance-comparison.md"
    text = document.read_text(encoding="utf-8").split(HEADING)[0].rstrip() + "\n\n" + "\n".join(lines)
    atomic_text(document, text)
    atomic_text(ROOT / "variance-comparison.md", text)
    print("Validated binary-success variances for 6 policies and 90 policy/seed groups.")
    for r in summaries:
        print(r["policy"], f"within-seed change {r['within_seed_variance_change_pct']:+.2f}%")


if __name__ == "__main__":
    main()
