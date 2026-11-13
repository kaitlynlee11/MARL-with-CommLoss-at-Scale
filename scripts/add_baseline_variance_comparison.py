"""Append all three baseline variance changes to the scheduler comparison report."""
import csv
import io
from pathlib import Path
from statistics import mean, variance

from run_adaptive_seeds import atomic_text

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scheduler-heldout500-20261112"
HEADING = "## Baseline policy variance changes: 200 versus 500 episodes"


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    old = read(REPO / "results/baseline-agent-spacing04-15-seeds-20261109/within-seed-episode-variation.csv")
    new = read(REPO / "results/baseline-heldout500-20261112/within-seed-episode-variation.csv")
    assert {(r['training_seed'],r['condition']) for r in old} == {(r['training_seed'],r['condition']) for r in new}
    assert len(old) == len(new) == 45
    rows = []
    for policy in ("full", "fixed_5", "none"):
        before = [r for r in old if r["condition"] == policy]
        after = [r for r in new if r["condition"] == policy]
        assert len(before) == len(after) == 15
        assert {int(r['episodes']) for r in before} == {200}
        assert {int(r['episodes']) for r in after} == {500}
        for metric in ("task_return", "coverage"):
            a,b = (mean(float(r[metric+"_variance"]) for r in subset) for subset in (before,after))
            c,d = (variance(float(r[metric+"_mean"]) for r in subset) for subset in (before,after))
            rows.append(dict(policy=policy, metric=metric, mean_within_seed_variance_200=a,
                mean_within_seed_variance_500=b, within_seed_variance_change_pct=100*(b-a)/a,
                across_seed_variance_200=c, across_seed_variance_500=d,
                across_seed_variance_change_pct=100*(d-c)/c))
    # None is already present as initial_none in the scheduler comparison.
    original = read(ROOT / "variance-summary.csv")
    for row in rows:
        if row["policy"] == "none":
            reference = next(r for r in original if r["group"] == "initial_none" and r["metric"] == row["metric"])
            assert all(abs(float(reference[k])-row[k]) < 1e-10 for k in row if k not in ("policy","metric"))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT / "baseline-variance-summary.csv",stream.getvalue())
    lines = [HEADING, "",
        "The full, fixed_5 and none baselines use the same 15 training seeds (42–56), the original 200 scenarios and expanded 500 scenarios as the scheduler studies, with unchanged checkpoints, zero loss, and 0.4-unit minimum agent-start and landmark spacing. Baseline none is the same policy and evaluation data labeled initial_none above; it is repeated here to show all three baselines together.", "",
        "### Average within-seed episode variance", "",
        "Changes use the same definition as above: average the 15 seed-specific sample variances, then compute 100 × (variance500 − variance200) / variance200.", "",
        "| Baseline | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |",
        "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['metric']} | {r['mean_within_seed_variance_200']:.6f} | {r['mean_within_seed_variance_500']:.6f} | {r['within_seed_variance_change_pct']:+.2f}% |")
    lines += ["", "### Variance across the 15 seed-level means", "",
        "| Baseline | Metric | Across-seed variance: 200 | Across-seed variance: 500 | Change |",
        "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['metric']} | {r['across_seed_variance_200']:.8f} | {r['across_seed_variance_500']:.8f} | {r['across_seed_variance_change_pct']:+.2f}% |")
    lines += ["", "Full-precision baseline summaries: `results/scheduler-heldout500-20261112/baseline-variance-summary.csv`. All 45 per-seed baseline changes: `results/baseline-heldout500-20261112/within-seed-variance-change-200-vs-500.csv`. Source episode statistics: the `within-seed-episode-variation.csv` files in the original and expanded baseline study directories.", ""]
    document = REPO / "docs/scheduler-holdout-500-variance-comparison.md"
    text = document.read_text(encoding="utf-8").split(HEADING)[0].rstrip() + "\n\n" + "\n".join(lines)
    atomic_text(document,text)
    atomic_text(ROOT / "variance-comparison.md",text)
    from add_success_variance_comparison import main as add_success
    add_success()
    print("Added all three baseline policies; verified none matches initial_none exactly.")


if __name__ == "__main__":
    main()
