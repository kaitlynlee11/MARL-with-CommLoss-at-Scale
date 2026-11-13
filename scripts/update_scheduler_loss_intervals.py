"""Add saved seed intervals and task-return means to the four-policy report."""
import json
from pathlib import Path

from run_adaptive_seeds import atomic_text, seed_stats, write_json

REPO = Path(__file__).resolve().parents[1]


def cell(stat, percentage=False):
    lo, hi = stat["seed_bootstrap_95pct"]
    assert lo <= hi
    def number(value):
        return f"{value:.2%}" if percentage else f"{value:.2f}"
    return f"{number(stat['mean'])} [{number(lo)}, {number(hi)}]"


def main():
    root = REPO / "results/scheduler-loss-comparison-20261110"
    doc = REPO / "docs/scheduler-loss-comparison-results.md"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "no-communication-reference.json").read_text(encoding="utf-8"))
    assert summary["complete"]
    task_stats = []
    conditions = {}
    for condition in summary["conditions"]:
        loss, method = condition["loss"], condition["condition"]
        rows = [r for r in summary["per_seed"] if r["loss"] == loss and r["condition"] == method]
        assert len(rows) == 15 and {r["training_seed"] for r in rows} == set(range(42, 57))
        # Keep saved intervals for other metrics. Use the same bootstrap method
        # and saved row order for the newly summarized task return.
        stat = seed_stats([r["task_return"] for r in rows])
        condition["metrics"]["task_return"] = stat
        conditions[loss, method] = condition["metrics"]
        task_stats.append({"loss": loss, "condition": method, **stat})
    assert len(reference["per_seed"]) == 15
    none_stat = seed_stats([r["task_return"] for r in reference["per_seed"]])
    reference["metrics_at_each_loss_rate"]["task_return"] = none_stat
    for loss in (.1, .25, .5):
        conditions[loss, "none"] = reference["metrics_at_each_loss_rate"]
        task_stats.append({"loss": loss, "condition": "none", **none_stat})

    lines = ["## Performance with 95% seed bootstrap intervals", "",
        "Every metric below is **mean [95% seed bootstrap interval]**, with equal weight for each of 15 independent training seeds. Intervals use the percentile bootstrap of seed-level episode means (20,000 resamples; analysis RNG seed 20261110). They do not treat the 3,000 evaluation episodes as independent training replicates. The `none` rows reuse the same loss-invariant baseline estimates at each loss rate.", "",
        "Task return is the mean cumulative base task reward over the 100-step episode, using the baseline-matched `legacy-distance-v0` scoring: `-0.5 * sum(nearest-agent distance to each landmark) - overlapping_agent_pairs / 3` per step. It excludes adaptive training bonuses and packet penalties; higher (less negative) is better.", "",
        "| Loss | Policy | Seeds | Success | Coverage | Task return | Collision step rate | Episodes with collisions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for loss in (.1, .25, .5):
        for method in ("adaptive", "full", "fixed_5", "none"):
            m = conditions[loss, method]
            values = [cell(m[key], percent) for key, percent in
                      [("success_rate", True), ("final_coverage", True), ("task_return", False),
                       ("collision_step_rate", True), ("collision_episode_fraction", True)]]
            lines.append(f"| {loss:.0%} | {method} | 15 | " + " | ".join(values) + " |")
    lines += ["", "## Packet counts with 95% seed bootstrap intervals", "",
        "Counts are directed packets per episode; completed packets are the delivered packets. The fixed schedules have constant attempt counts. Their delivery counts also have degenerate seed intervals because all policies use the same scenario-indexed channel draws across training seeds; these seed intervals do not measure uncertainty over new channel realizations.", "",
        "| Loss | Policy | Delivered packets/episode | Attempted packets/episode |",
        "| --- | --- | --- | --- |"]
    for loss in (.1, .25, .5):
        for method in ("adaptive", "full", "fixed_5", "none"):
            m = conditions[loss, method]
            lines.append(f"| {loss:.0%} | {method} | {cell(m['delivered_links'])} | {cell(m['attempted_links'])} |")
    lines.append("")
    text = doc.read_text(encoding="utf-8")
    start_marker = "## Performance with 95% seed bootstrap intervals" if "## Performance with 95% seed bootstrap intervals" in text else "| Loss | Policy | Seeds | Success |"
    start = text.index(start_marker)
    end = text.index("Means weight independent training seeds equally.", start)
    text = text[:start] + "\n".join(lines) + "\n" + text[end:]
    text = text.replace("summary.json includes sample SDs,", "summary.json (and no-communication-reference.json for `none`) includes task-return means, sample SDs,")
    write_json(root / "summary.json", summary)
    write_json(root / "no-communication-reference.json", reference)
    write_json(root / "task-return-seed-bootstrap.json", {
        "resamples": 20000, "analysis_rng_seed": 20261110, "unit": "independent training seed",
        "method": "percentile bootstrap of seed-level mean task returns",
        "reward_version": "legacy-distance-v0", "conditions": task_stats})
    atomic_text(doc, text)
    atomic_text(root / "report.md", text)
    assert len([line for line in text.splitlines() if line.startswith("| ") and " | 15 | " in line]) == 12
    assert sum(line.startswith("| ") and "[" in line for line in text.splitlines()) == 24
    print("Updated 12 policy/loss rows with task-return means and seed intervals; packet intervals shown separately.")


if __name__ == "__main__":
    main()
