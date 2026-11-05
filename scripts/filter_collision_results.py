"""Recalculate both collision-study stages conditional on collision-free starts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from limited_comm.collision_study import collision_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/collision-zero-loss-20261104'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path,default=Path('docs/collision-zero-loss-safe-start-results.md'))
    args=parser.parse_args()
    source=args.source.resolve()
    output=(args.output or source/'safe-starts').resolve()
    records={phase:[json.loads(line) for line in (source/phase/'episodes.jsonl').read_text().splitlines() if line.strip()]
             for phase in ['frozen','retrained']}
    excluded_sets=[]
    retained_sets=[]
    for phase,rows in records.items():
        for label in ['adaptive','fixed_5','full']:
            subset=[r for r in rows if r['condition']==label]
            assert len(subset)==200 and len({r['seed'] for r in subset})==200
            excluded_sets.append({r['seed'] for r in subset if r['initial_collision']})
            retained_sets.append({r['seed'] for r in subset if not r['initial_collision']})
            assert all(r['active_steps']==0 and r['attempted_links']==0 and not r['final_success']
                       for r in subset if r['initial_collision'])
    assert all(s==excluded_sets[0] for s in excluded_sets)
    assert all(s==retained_sets[0] for s in retained_sets)
    excluded=sorted(excluded_sets[0])
    retained=sorted(retained_sets[0])
    assert len(excluded)==30 and len(retained)==170
    output.mkdir(parents=True,exist_ok=False)
    manifest={'source':str(source),'excluded_seeds':excluded,'retained_seeds':retained,
              'original_scenarios_per_condition':200,'excluded_scenarios_per_condition':30,
              'retained_scenarios_per_condition':170,'rule':'exclude initial_collision == true; identical seed exclusion across all conditions and both stages',
              'fixed_packet_denominator':600,'loss':0.,'retraining_performed':False,
              'source_episode_sha256':{phase:hashlib.sha256((source/phase/'episodes.jsonl').read_bytes()).hexdigest() for phase in records}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'excluded-seeds.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    studies={}
    for phase,rows in records.items():
        destination=output/phase
        destination.mkdir()
        kept=[r for r in rows if not r['initial_collision']]
        assert len(kept)==510
        summaries={}
        for label in ['adaptive','fixed_5','full']:
            subset=[r for r in kept if r['condition']==label]
            previous=[r for r in rows if r['condition']==label]
            summaries[label]=collision_summary(subset)
            assert summaries[label]['initial_collision_fraction']==0
            assert sum(r['attempted_links'] for r in subset)==sum(r['attempted_links'] for r in previous)
            assert sum(r['active_steps'] for r in subset)==sum(r['active_steps'] for r in previous)
            assert sum(r['final_success'] for r in subset)==sum(r['final_success'] for r in previous)
            assert np.isclose(summaries[label]['attempts_per_active_step'],collision_summary(previous)['attempts_per_active_step'])
            (destination/f'{label}.episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset),encoding='utf-8')
        (destination/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')
        original=json.loads((source/phase/'summary.json').read_text())
        studies[phase]={**original,'scope':original['scope']+'; recalculated conditional on collision-free reset',
                        'episodes_per_condition':170,'seed_start':None,'scenario_seeds':retained,
                        'excluded_reset_collision_seeds':excluded,'conditions':summaries,
                        'initial_collision_rule':'reset-collision scenarios excluded; later collisions still terminate and fail'}
        (destination/'summary.json').write_text(json.dumps(studies[phase],indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps({'cohort':manifest,**studies},indent=2),encoding='utf-8')
    lines=['# Collision-terminated comparison with reset collisions excluded','',
           'Recalculated November 4, 2026 from saved episode records. The same **30 reset-collision scenarios (15%)** were removed from all three schedulers in both stages. This leaves **170 matched scenarios per condition per stage**, or 1,020 retained episode records. No policies were retrained, no trajectories rerun, and no replacement scenarios added.','',
           'Success still requires collision-free survival through step 100 and full landmark coverage at that final step. Collisions after reset remain immediate failures. Packet loss is zero. The communication denominator remains **600 packets per retained scheduled 100-step episode**.','',
           '| Stage | Scheduler | Success (95% Wilson interval) | Any collision after reset | Packets / 600 | Packets/episode | Attempts/active step | Mean steps to termination |',
           '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for phase,study in studies.items():
        for label,m in study['conditions'].items():
            lo,hi=m['success_95pct_wilson']
            lines.append(f"| {phase} | {label} | {m['success_rate']:.2%} ({lo:.2%}-{hi:.2%}) | {m['collision_episode_fraction']:.2%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {m['attempts_per_active_step']:.3f} | {m['mean_active_steps']:.2f} |")
    lines+=['','## Counts and termination details','',
            '| Stage | Scheduler | Successful episodes | Collision episodes | Horizon survivors | Median steps | Min-max steps | Mean steps among collision failures |',
            '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for phase,study in studies.items():
        for label,m in study['conditions'].items():
            subset=[r for r in records[phase] if r['condition']==label and not r['initial_collision']]
            failed_steps='undefined' if m['mean_steps_before_collision'] is None else f"{m['mean_steps_before_collision']:.2f}"
            lines.append(f"| {phase} | {label} | {sum(r['final_success'] for r in subset)}/170 | {sum(r['any_collision'] for r in subset)}/170 | {sum(r['active_steps']==100 and not r['any_collision'] for r in subset)}/170 | {m['median_active_steps']:.1f} | {m['min_active_steps']}-{m['max_active_steps']} | {failed_steps} |")
    lines+=['','## What changed','',
            '- All excluded episodes had zero active steps, zero packets and no success. Removing them preserves the total successful episodes, attempted packets and executed steps, while changing the number of evaluated episodes from 200 to 170.',
            '- Consequently, success rate, mean packets per episode, packets/600, and mean executed steps increase by a factor of 200/170 relative to the original tables. Collision fractions are recomputed after subtracting the 30 initial failures from each condition.',
            '- Attempts per active step is unchanged: total attempts divided by total executed steps has the same numerator and denominator after zero-step episodes are removed.',
            '- The previous 85% ceiling caused by initial collisions no longer applies to this filtered cohort. These are results conditional on a collision-free start, not results for the original unrestricted reset distribution.',
            '- Policies and checkpoint selections are unchanged. The frozen adaptive policy was originally trained with 25% loss; all evaluations here and the collision-aware retraining used zero loss. One training seed per condition; Wilson intervals describe episode uncertainty only.','',
            '## Artifacts','']
    report=args.report.resolve()
    def relative(path):
        import os
        return Path(os.path.relpath(path,report.parent)).as_posix()
    lines += [f'- [Filtered summaries]({relative(output/"summary.json")})',
              f'- [Excluded scenario seeds]({relative(output/"excluded-seeds.json")})',
              f'- [Frozen retained episode records]({relative(output/"frozen/episodes.jsonl")})',
              f'- [Retrained retained episode records]({relative(output/"retrained/episodes.jsonl")})',
              f'- [Original all-start comparison]({relative(source/"report.md")})',
              f'- [Original methodology]({relative(Path("docs/collision-zero-loss-study.md").resolve())})','',
              'The original files and replays are preserved. Filtering changes aggregate statistics; it does not change retained trajectories. The existing first-scenario comparison replays use seed 400000, which belongs to the retained cohort.','']
    assert report.exists() is False,'Choose a new report path rather than overwriting an existing recalculation.'
    report.write_text('\n'.join(lines),encoding='utf-8')
    (output/'report.md').write_text('\n'.join(lines[:lines.index('## Artifacts')]),encoding='utf-8')
    print(f'Excluded {len(excluded)} identical scenario seeds across all six condition/stage groups; retained 1,020 records.')
    for phase,study in studies.items():
        for label,m in study['conditions'].items():
            print(phase,label,json.dumps({k:m[k] for k in ['success_rate','collision_episode_fraction','attempted_packet_ratio','attempted_links','attempts_per_active_step','mean_active_steps']}))
    print(report)


if __name__=='__main__':
    main()
