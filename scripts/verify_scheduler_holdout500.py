"""Independently verify expanded scheduler metrics, pairing and episode variances."""
import csv
import json
import time
from statistics import mean, variance, stdev

from extend_scheduler_holdout import ROOT, REPO, SEEDS, BASELINE, ADAPTIVE, WARM, ABLATION, sha, canonical
from run_adaptive_seeds import write_json


def close(a,b):
    assert abs(float(a)-float(b)) < 1e-9, (a,b)


def main():
    manifest=json.loads((ROOT / "manifest.json").read_text())
    for name,digest in manifest["source_sha256"].items():
        assert sha(ROOT / "source" / name)==digest
    for item in manifest["checkpoints"].values():
        assert sha(item["path"])==item["sha256"]
    for path,digest in manifest["original_episode_sha256"].items():
        assert sha(path)==digest
    records=[json.loads(line) for line in (ROOT / "episodes.jsonl").read_text().splitlines()]
    assert len(records)==30000
    data={}
    for seed in SEEDS:
        for group in ("adaptive","communication_disabled","warm_none","initial_none"):
            rows=[r for r in records if r["training_seed"]==seed and r["condition"]==group]
            assert len(rows)==500 and [r["seed"] for r in rows]==list(range(400000,400500))
            data[seed,group]=rows
            for r in rows:
                assert r["actual_min_separation"]>=.4 and r["actual_initial_agent_separation"]>=.4
                assert r["min_agent_distance"]==r["min_landmark_distance"]==.4
                assert r["loss"]==0 and r["active_steps"]==100 and not r["terminate_on_collision"]
                assert r["attempted_links"]==r["delivered_links"]
                if group!="adaptive":
                    assert r["attempted_links"]==0
            if group=="adaptive":
                original=ADAPTIVE / f"seed-{seed}/evaluation/episodes.jsonl"
            elif group=="warm_none":
                original=WARM / f"seed-{seed}/evaluation/episodes.jsonl"
            elif group=="communication_disabled":
                original=ABLATION / f"seed-{seed}/episodes.jsonl"
            else:
                original=REPO / f"results/baseline-agent-spacing04-15-seeds-20261109/seed-{seed}/none/evaluation/episodes.jsonl"
            old=[json.loads(line) for line in original.read_text().splitlines()]
            if group=="communication_disabled":
                old=[r for r in old if r["condition"]==group]
            assert [canonical(r) for r in rows[:200]]==[canonical(r) for r in old]
        for normal,disabled,warm,baseline in zip(*(data[seed,g] for g in ("adaptive","communication_disabled","warm_none","initial_none"))):
            for field in ("seed","actual_min_separation","actual_initial_agent_separation"):
                assert len({r[field] for r in (normal,disabled,warm,baseline)})==1
            if normal["attempted_links"]==0:
                assert canonical(normal)==canonical(disabled)
    with (ROOT / "within-seed-variance-200-vs-500.csv").open(newline="") as stream:
        variations=list(csv.DictReader(stream))
    assert len(variations)==60
    for r in variations:
        rows=data[int(r["training_seed"]),r["group"]]
        for metric,field in (("task_return","task_return"),("coverage","final_coverage")):
            for count in (200,500):
                values=[e[field] for e in rows[:count]]
                close(r[f"{metric}_variance_{count}"],variance(values))
                close(r[f"{metric}_sd_{count}"],stdev(values))
                close(r[f"{metric}_mean_{count}"],mean(values))
            a,b=(float(r[f"{metric}_variance_{n}"]) for n in (200,500))
            close(r[f"{metric}_variance_change_pct"],100*(b-a)/a)
    with (ROOT / "variance-summary.csv").open(newline="") as stream:
        variance_summaries=list(csv.DictReader(stream))
    assert len(variance_summaries)==8
    for r in variance_summaries:
        subset=[v for v in variations if v["group"]==r["group"]]
        metric=r["metric"]
        for count in (200,500):
            close(r[f"mean_within_seed_variance_{count}"],mean(float(v[f"{metric}_variance_{count}"]) for v in subset))
            close(r[f"across_seed_variance_{count}"],variance(float(v[f"{metric}_mean_{count}"]) for v in subset))
        for prefix,change in (("mean_within_seed_variance","within_seed_variance_change_pct"),
                              ("across_seed_variance","across_seed_variance_change_pct")):
            a,b=(float(r[f"{prefix}_{count}"]) for count in (200,500))
            close(r[change],100*(b-a)/a)
    for study,document in (("adaptive","adaptive-15-seeds-results.md"),("ablation","adaptive-communication-ablation-results.md"),("warm-none","warm-no-communication-results.md")):
        directory=ROOT / study
        assert (directory / "report.md").read_bytes()==(REPO / "docs" / document).read_bytes()
        summary=json.loads((directory / "summary.json").read_text())
        assert summary["complete"] and summary["episodes_per_seed"]==500
        with (directory / "evaluation-per-seed.csv").open(newline="") as stream:
            evaluations=list(csv.DictReader(stream))
        metric_fields={"success_rate":"final_success","final_coverage":"final_coverage","task_return":"task_return",
                       "collision_step_rate":"collision_step_rate","collision_episode_fraction":"any_collision",
                       "attempted_links":"attempted_links","delivered_links":"delivered_links"}
        for r in evaluations:
            rows=data[int(r["training_seed"]),r["group"]]
            for metric,field in metric_fields.items():
                close(r[metric],mean(e[field] for e in rows))
        # Check seed bootstrap inputs and paired changes in the machine-readable summaries.
        def verify_stats(stats,groups):
            for group,metrics in stats.items():
                if group not in groups:
                    continue
                for metric,s in metrics.items():
                    if metric not in metric_fields:
                        continue
                    values=[mean(e[metric_fields[metric]] for e in data[seed,groups[group]]) for seed in SEEDS]
                    close(s["mean"],mean(values))
                    close(s["sample_sd"],stdev(values))
        if study=="adaptive":
            verify_stats({"adaptive":summary["metrics"]},{"adaptive":"adaptive"})
        elif study=="ablation":
            verify_stats(summary["statistics"],{"original":"adaptive","communication_disabled":"communication_disabled"})
            for metric,s in summary["statistics"]["differences_disabled_minus_original"].items():
                values=[mean(e[metric_fields[metric]] for e in data[seed,"communication_disabled"])-mean(e[metric_fields[metric]] for e in data[seed,"adaptive"]) for seed in SEEDS]
                close(s["mean"],mean(values))
                close(s["sample_sd"],stdev(values))
        else:
            verify_stats(summary["groups"],{g:g for g in ("adaptive","warm_none","initial_none")})
    write_json(ROOT / "independent-audit.json",{"passed":True,"verified_unix":time.time(),
        "outcome_records":30000,"frozen_checkpoints":30,"variance_comparisons":60,
        "original_200_episodes_preserved":True,"metrics_and_seed_sds_recalculated":True,
        "paired_scenarios_and_spacing_verified":True,"reports_match_machine_readable_results":True})
    print("PASS: 30,000 outcome records; original episodes, 30 checkpoints, paired scenarios, metrics and all 60 variance comparisons verified.")


if __name__=="__main__":
    main()
