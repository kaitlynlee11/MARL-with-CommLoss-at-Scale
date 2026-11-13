"""Audit and report the three expanded, paired 500-episode scheduler studies."""
import csv
import io
import json
from pathlib import Path
from statistics import mean, variance, stdev
import sys
import time

from extend_scheduler_holdout import (ROOT, REPO, ADAPTIVE, ABLATION, WARM, BASELINE,
                                     SEEDS, sha, canonical, old_records)
from run_adaptive_seeds import atomic_text, read_json, write_json, targets_met

GROUPS = ("adaptive", "communication_disabled", "warm_none", "initial_none")
METRICS = ("success_rate", "final_coverage", "task_return", "collision_step_rate",
           "collision_episode_fraction", "attempted_links", "delivered_links")


def write_csv(path, rows):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path,stream.getvalue())


def main():
    sys.path.insert(0,str(ROOT / "source"))
    from limited_comm.evaluate import summarize
    from run_adaptive_seeds import aggregate as adaptive_aggregate
    from evaluate_adaptive_ablation import aggregate as ablation_aggregate
    from train_warm_no_comm import aggregate as warm_aggregate
    manifest=read_json(ROOT / "manifest.json")
    baseline_manifest=read_json(BASELINE / "manifest.json")
    for seed in SEEDS:
        reference=baseline_manifest["checkpoints"][f"{seed}/none"]
        assert sha(reference["path"])==reference["sha256"]
    for relative,digest in manifest["source_sha256"].items():
        assert sha(ROOT / "source" / relative)==digest
    for item in manifest["checkpoints"].values():
        assert sha(item["path"])==item["sha256"]
    for path,digest in manifest["original_episode_sha256"].items():
        assert sha(path)==digest
    data={}
    summaries={}
    variance_rows=[]
    for seed in SEEDS:
        for group in GROUPS:
            if group=="initial_none":
                path=BASELINE / f"seed-{seed}/none/episodes.jsonl"
                records=[{"training_seed":seed,"condition":group,**canonical(json.loads(line))}
                         for line in path.read_text().splitlines()]
            else:
                path=ROOT / f"evaluations/{group}/seed-{seed}/episodes.jsonl"
                records=[json.loads(line) for line in path.read_text().splitlines()]
                assert [canonical(r) for r in records[:200]]==old_records(seed,group)
            assert len(records)==500 and [r["seed"] for r in records]==list(range(400000,400500))
            for r in records:
                assert r["min_agent_distance"]==r["min_landmark_distance"]==.4
                assert r["actual_initial_agent_separation"]>=.4 and r["actual_min_separation"]>=.4
                assert r["loss"]==0 and r["active_steps"]==100 and not r["terminate_on_collision"]
                assert not r["initial_collision"] and r["task_reward_version"]=="legacy-distance-v0"
                assert r["attempted_links"]==r["delivered_links"]
                if group!="adaptive":
                    assert r["attempted_links"]==r["payload_bytes"]==0
            data[seed,group]=records
            metrics={**summarize(records),"collision_episode_fraction":mean(r["any_collision"] for r in records)}
            summaries[seed,group]=metrics
            row={"training_seed":seed,"group":group,"episodes_200":200,"episodes_500":500}
            for metric,field in (("task_return","task_return"),("coverage","final_coverage")):
                for count in (200,500):
                    values=[r[field] for r in records[:count]]
                    row[f"{metric}_mean_{count}"]=mean(values)
                    row[f"{metric}_variance_{count}"]=variance(values)
                    row[f"{metric}_sd_{count}"]=stdev(values)
                previous=row[f"{metric}_variance_200"]
                row[f"{metric}_variance_change_pct"]=100*(row[f"{metric}_variance_500"]-previous)/previous
            variance_rows.append(row)
        for rows in zip(*(data[seed,g] for g in GROUPS)):
            for field in ("seed","actual_initial_agent_separation","actual_min_separation"):
                assert len({r[field] for r in rows})==1, (seed,field)
            normal,disabled=rows[:2]
            if normal["attempted_links"]==0:
                assert canonical(normal)==canonical(disabled)
    write_csv(ROOT / "within-seed-variance-200-vs-500.csv",variance_rows)
    atomic_text(ROOT / "episodes.jsonl","".join(json.dumps(r)+"\n" for records in data.values() for r in records))
    adaptive_rows=[]
    ablation_rows=[]
    warm_rows=[]
    paired=[]
    for seed in SEEDS:
        normal=data[seed,"adaptive"]
        disabled=data[seed,"communication_disabled"]
        a,b=summaries[seed,"adaptive"],summaries[seed,"communication_disabled"]
        row=read_json(ADAPTIVE / f"seed-{seed}/result.json")
        row.update(a)
        row.update(heldout_targets_met=targets_met(a["success_rate"],a["attempted_links"]),
                   max_packets_in_episode=max(r["attempted_links"] for r in normal),
                   episodes_at_or_above_120_packets=sum(r["attempted_links"]>=120 for r in normal))
        for policy in ("full","none","fixed_5"):
            baseline_records=[json.loads(line) for line in (BASELINE / f"seed-{seed}/{policy}/episodes.jsonl").read_text().splitlines()]
            assert len(baseline_records)==500
            row["comparisons"][policy]={"baseline_metrics":summarize(baseline_records),
                "success_to_failure":sum(x["final_success"] and not y["final_success"] for x,y in zip(baseline_records,normal)),
                "failure_to_success":sum(not x["final_success"] and y["final_success"] for x,y in zip(baseline_records,normal))}
        adaptive_rows.append(row)
        pairs=[{"training_seed":seed,"scenario_seed":x["seed"],
                "original_success":x["final_success"],"disabled_success":y["final_success"],
                "original_packets":x["attempted_links"],"success_changed":x["final_success"]!=y["final_success"],
                **{f"difference_{m}":y[m]-x[m] for m in ("task_return","collision_step_rate","final_coverage","collision_pairs")}}
               for x,y in zip(normal,disabled)]
        paired.extend(pairs)
        ablation_rows.append({"training_seed":seed,"original":a,"communication_disabled":b,
            "differences_disabled_minus_original":{m:b[m]-a[m] for m in METRICS},"paired_scenarios":500,
            "original_records_reproduced":True,"original_communicating_episodes":sum(r["attempted_links"]>0 for r in normal),
            "success_to_failure":sum(p["original_success"] and not p["disabled_success"] for p in pairs),
            "failure_to_success":sum(not p["original_success"] and p["disabled_success"] for p in pairs),
            "collision_step_rate_changed_episodes":sum(p["difference_collision_step_rate"]!=0 for p in pairs),
            "task_return_changed_episodes":sum(abs(p["difference_task_return"])>1e-9 for p in pairs)})
        row=read_json(WARM / f"seed-{seed}/result.json")
        row["groups"]={g:summaries[seed,g] for g in ("initial_none","warm_none","adaptive")}
        row["differences"]={f"warm_none_minus_{g}":{m:row["groups"]["warm_none"][m]-row["groups"][g][m] for m in METRICS}
                            for g in ("initial_none","adaptive")}
        warm_rows.append(row)
    studies=(("adaptive",adaptive_rows,adaptive_aggregate,"adaptive-15-seeds-results.md",("adaptive",),ADAPTIVE),
             ("ablation",ablation_rows,ablation_aggregate,"adaptive-communication-ablation-results.md",("adaptive","communication_disabled"),ABLATION),
             ("warm-none",warm_rows,warm_aggregate,"warm-no-communication-results.md",("initial_none","warm_none","adaptive"),WARM))
    variance_summary=[]
    for group in GROUPS:
        subset=[r for r in variance_rows if r["group"]==group]
        for metric in ("task_return","coverage"):
            previous=mean(r[f"{metric}_variance_200"] for r in subset)
            current=mean(r[f"{metric}_variance_500"] for r in subset)
            across200=variance(r[f"{metric}_mean_200"] for r in subset)
            across500=variance(r[f"{metric}_mean_500"] for r in subset)
            variance_summary.append({"group":group,"metric":metric,"mean_within_seed_variance_200":previous,
                "mean_within_seed_variance_500":current,"within_seed_variance_change_pct":100*(current-previous)/previous,
                "across_seed_variance_200":across200,"across_seed_variance_500":across500,
                "across_seed_variance_change_pct":100*(across500-across200)/across200})
    write_csv(ROOT / "variance-summary.csv",variance_summary)
    def variance_change_lines(groups):
        lines=["## Variance change: 200 versus 500 held-out episodes","",
            "Each seed's episode variance is a sample variance (denominators 199 and 499). This table first averages those variances equally across the 15 seeds, then reports 100 × (variance500 − variance200) / variance200. It is not a change in standard error or a mean of seed-specific percentage changes. Coverage variance uses fractions on the 0–1 scale; task-return variance uses squared return units. Individual seed changes can have either sign. More evaluation episodes improve the estimate; they do not inherently reduce outcome variance.","",
            "| Group | Metric | Mean within-seed variance: 200 | Mean within-seed variance: 500 | Change |",
            "| --- | --- | --- | --- | --- |"]
        for row in variance_summary:
            if row["group"] in groups:
                lines.append(f"| {row['group']} | {row['metric']} | {row['mean_within_seed_variance_200']:.6f} | {row['mean_within_seed_variance_500']:.6f} | {row['within_seed_variance_change_pct']:+.2f}% |")
        return lines
    for name,rows,aggregate,document,groups,origin in studies:
        directory=ROOT / name
        directory.mkdir(exist_ok=True)
        archive=directory / "original-200-episode-report.md"
        if not archive.exists():
            atomic_text(archive,(origin / "report.md").read_text(encoding="utf-8"))
        aggregate(directory,rows,SEEDS,evaluation_episodes=500)
        summary=read_json(directory / "summary.json")
        summary.update(episodes_per_seed=500,scenario_seeds=[400000,400499],original_episodes_per_seed=200,
                       additional_episodes_per_seed=300,shared_manifest="../manifest.json")
        write_json(directory / "summary.json",summary)
        evaluation_rows=[{"training_seed":seed,"group":g,**summaries[seed,g]} for seed in SEEDS for g in groups]
        write_csv(directory / "evaluation-per-seed.csv",evaluation_rows)
        write_csv(directory / "within-seed-variance-200-vs-500.csv",[r for r in variance_rows if r["group"] in groups])
        atomic_text(directory / "episodes.jsonl","".join(json.dumps(r)+"\n" for seed in SEEDS for g in groups for r in data[seed,g]))
        text=(directory / "report.md").read_text(encoding="utf-8")
        text=text.replace("Artifacts: per-seed.csv, summary.json, seed-N/training/learning.jsonl, checkpoint/initializer hashes, confirmations, frozen evaluation/episodes.jsonl, selection.json and result.json. The live progress.html also opens directly from disk and refreshes every five seconds.","")
        # Replace historical artifact descriptions with paths for the expanded evaluation.
        text="\n".join(line for line in text.splitlines() if not line.startswith(("Artifacts:","Individual outcomes:","Full plan and initializer/adaptive-checkpoint hashes:")))
        text=text.replace("Suppressed gate decisions describe the raw transmit choices under that counterfactual history, not the original policy's packet count.","")
        if name=="warm-none":
            budget=["## Seed-specific optimization budgets","",
                    "Training budgets and initialization audits are unchanged from the archived training study. These totals exclude held-out evaluation steps.","",
                    "| Seed | Additional world steps | PPO updates | Optimizer steps | Total work including initializer |",
                    "| --- | --- | --- | --- | --- |"]
            for r in rows:
                budget.append(f"| {r['training_seed']} | {r['control_world_steps']:,} | {r['ppo_updates']:,} | {r['optimizer_steps']:,} | {r['total_training_work_including_initializer']:,} |")
            text=text.replace("## Seed-specific optimization budgets","\n".join(budget))
        extra=["", "## Expanded evaluation provenance","",
            f"This is the completed 500-episode held-out evaluation. The original 200 scenarios (400000–400199) are retained, and 300 scenarios (400200–400499) are added for each training seed. Frozen checkpoints and checkpoint selection are unchanged; no retraining was performed. Original training artifacts, initialization audits and 200-episode results remain at `{origin.relative_to(REPO).as_posix()}/`.","",
            f"Current artifacts: `{directory.relative_to(REPO).as_posix()}/summary.json`, `per-seed.csv`, `evaluation-per-seed.csv`, `episodes.jsonl`, and `within-seed-variance-200-vs-500.csv`. Shared checkpoint/source hashes, raw evaluations and verification: `results/scheduler-heldout500-20261112/`. Initial-none and other baseline references use the completed 500-episode baseline evaluation. The adaptive results in the ablation and warm-none comparison are the same outcomes, not independent reruns.",""]
        if name=="warm-none":
            extra += ["## Individual held-out evaluations","","| Seed | Group | Success | Coverage | Task return | Collision step rate | Packets/episode |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in evaluation_rows:
                extra.append(f"| {r['training_seed']} | {r['group']} | {r['success_rate']:.2%} | {r['final_coverage']:.2%} | {r['task_return']:.4f} | {r['collision_step_rate']:.3%} | {r['attempted_links']:.4f} |")
            extra.append("")
        extra += ["## Within-seed episode variance and SD (500 episodes)","",
                  "Coverage SD is in percentage points. These are episode statistics within a training seed; the overall report's seed SDs describe variation of seed-level means.","",
                  "| Seed | Group | Task return variance | Task return SD | Coverage variance (0–1 scale) | Coverage SD (pp) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in variance_rows:
            if r["group"] in groups:
                extra.append(f"| {r['training_seed']} | {r['group']} | {r['task_return_variance_500']:.3f} | {r['task_return_sd_500']:.3f} | {r['coverage_variance_500']:.6f} | {100*r['coverage_sd_500']:.2f} |")
        extra += ["",*variance_change_lines(groups),""]
        text=text.rstrip()+"\n\n"+"\n".join(extra)
        atomic_text(directory / "report.md",text)
        atomic_text(REPO / "docs" / document,text)
    atomic_text(ROOT / "ablation/paired-episodes.jsonl","".join(json.dumps(r)+"\n" for r in paired))
    variance_report=["# Scheduler held-out evaluation: variance change from 200 to 500 episodes","",
        "Training seeds 42–56; exactly matched scenarios 400000–400499 across all groups; three agents and three landmarks; minimum pairwise radial agent-start and landmark spacing 0.4; zero loss; horizon 100; no collision termination. Training, validation, confirmation and checkpoint choices are unchanged. Initial none reuses the expanded baseline results. Adaptive normal results are shared between all three studies.","",
        *variance_change_lines(GROUPS),"","## Variance across the 15 seed-level means","",
        "This is a separate quantity: the sample variance of the 15 evaluation means. It is not within-seed episode variance.","",
        "| Group | Metric | Across-seed variance: 200 | Across-seed variance: 500 | Change |","| --- | --- | --- | --- | --- |"]
    for row in variance_summary:
        variance_report.append(f"| {row['group']} | {row['metric']} | {row['across_seed_variance_200']:.8f} | {row['across_seed_variance_500']:.8f} | {row['across_seed_variance_change_pct']:+.2f}% |")
    variance_report += ["","Full precision and all 60 seed/group comparisons: `results/scheduler-heldout500-20261112/within-seed-variance-200-vs-500.csv`. Policy-level summaries: `results/scheduler-heldout500-20261112/variance-summary.csv`.",""]
    atomic_text(REPO / "docs/scheduler-holdout-500-variance-comparison.md","\n".join(variance_report))
    atomic_text(ROOT / "variance-comparison.md","\n".join(variance_report))
    write_json(ROOT / "verification.json",{"passed":True,"training_seeds":15,"heldout_scenarios_per_seed":500,
        "new_rollouts":13500,"adaptive_episodes":7500,"disabled_episodes":7500,"warm_none_episodes":7500,
        "baseline_none_reference_episodes":7500,"old_outcomes_preserved":True,"scenario_pairing_verified":True,
        "spacing_verified":True,"zero_loss_verified":True,"frozen_checkpoints_verified":30,
        "original_source_and_episode_hashes_verified":True,"silent_originals_match_disabled_exactly":True,
        "within_seed_variance_comparisons":60,"baseline_none_checkpoints_verified":15,
        "baseline_manifest_sha256":sha(BASELINE / "manifest.json")})
    analysis=ROOT / "analysis-source"
    analysis.mkdir(exist_ok=True)
    for name in ("extend_scheduler_holdout.py","summarize_scheduler_holdout500.py","verify_scheduler_holdout500.py",
                 "run_adaptive_seeds.py","evaluate_adaptive_ablation.py","train_warm_no_comm.py"):
        atomic_text(analysis / name,(REPO / "scripts" / name).read_text(encoding="utf-8"))
    write_json(ROOT / "analysis-source-sha256.json",{p.name:sha(p) for p in analysis.glob("*.py")})
    status=read_json(ROOT / "status.json",{})
    status.update(state="complete",completed_condition_runs=45,episodes_ready=22500,
                  report_generation_complete=True,updated_unix=time.time())
    write_json(ROOT / "status.json",status)
    print("Audited and summarized 500 held-out scenarios per seed for all three studies.",flush=True)


if __name__=="__main__":
    main()
