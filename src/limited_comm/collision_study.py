"""Matched collision-terminated evaluation of frozen scheduler policies."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import evaluate_episode, summarize, wilson
from .mappo import load_policy
from .visualize import HTML, collect_replay


def collision_summary(rows, horizon=100):
    summary = summarize(rows)
    steps = sum(row['active_steps'] for row in rows)
    collided = [row for row in rows if row['any_collision']]
    initial = sum(row['initial_collision'] for row in rows)
    summary.update(
        collision_episode_fraction=len(collided)/len(rows),
        collision_episode_95pct_wilson=wilson(len(collided),len(rows)),
        initial_collision_fraction=initial/len(rows),
        success_upper_bound_given_initial_collisions=1-initial/len(rows),
        attempted_packet_ratio=summary['attempted_links']/(6*horizon),
        attempts_per_active_step=sum(row['attempted_links'] for row in rows)/steps if steps else None,
        mean_active_steps=steps/len(rows),
        median_active_steps=float(np.median([row['active_steps'] for row in rows])),
        min_active_steps=min(row['active_steps'] for row in rows),
        max_active_steps=max(row['active_steps'] for row in rows),
        mean_steps_before_collision=float(np.mean([row['active_steps'] for row in collided])) if collided else None,
        horizon_survival_fraction=sum(row['active_steps']==horizon and not row['any_collision'] for row in rows)/len(rows))
    return summary


def evaluate_study(paths, output, episodes=200, seed=400000, scope='existing frozen policies; no retraining', min_agent_distance=0.4):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    summaries, records, provenance, models = {}, [], {}, {}
    for label,schedule in [('adaptive','adaptive'),('fixed_5','fixed'),('full','full')]:
        path=Path(paths[label])
        model,checkpoint=load_policy(path)
        cfg=checkpoint['config']
        assert cfg['horizon']==100 and cfg['radius']==.6 and cfg.get('min_landmark_distance')==.4
        assert bool(model.spec.get('adaptive')) == (schedule=='adaptive')
        models[schedule]=model
        provenance[label]={'checkpoint':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'checkpoint_steps':checkpoint['steps'],'training_config':cfg}
        rows=[]
        with (output/f'{label}.episodes.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
            for index in range(episodes):
                row={'condition':label, **evaluate_episode(model,seed+index,schedule,period=5,loss=0.,horizon=100,
                      radius=.6,min_landmark_distance=.4,terminate_on_collision=True,min_agent_distance=min_agent_distance)}
                assert row['attempted_links']==row['delivered_links']
                assert 0<=row['active_steps']<=100
                assert not row['final_success'] or (row['active_steps']==100 and not row['any_collision'])
                if label=='full':
                    assert row['attempted_links']==6*row['active_steps']
                if label=='fixed_5':
                    assert row['attempted_links']==6*((row['active_steps']+4)//5)
                rows.append(row)
                stream.write(json.dumps(row)+'\n')
                if (index+1)%50==0:
                    print(f'{label}: {index+1}/{episodes}',flush=True)
        summaries[label]=collision_summary(rows)
        records.extend(rows)
        (output/'progress.json').write_text(json.dumps({'completed_conditions':list(summaries),
                         'summaries':summaries},indent=2),encoding='utf-8')
    initial_sets=[{r['seed'] for r in records if r['condition']==label and r['initial_collision']} for label in summaries]
    assert initial_sets[0]==initial_sets[1]==initial_sets[2]
    with (output/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row)+'\n' for row in records))
    result={'scope':scope,'loss':0.,'horizon':100,'agents':3,'radius':.6,'min_landmark_distance':.4,'min_agent_distance':min_agent_distance,
            'seed_start':seed,'episodes_per_condition':episodes,'fixed_packet_denominator':600,
            'initial_collision_rule':'immediate failure at step 0; no actions or packet attempts',
            'success_definition':'survive collision-free through step 100 and finish with full coverage',
            'conditions':summaries,'checkpoints':provenance}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    seeds={'comparison.html':seed}
    safe=next((s for s in range(seed,seed+episodes) if s not in initial_sets[0]),None)
    if safe is not None and safe!=seed:
        seeds['safe-start.html']=safe
    successes=[{r['seed'] for r in records if r['condition']==label and r['final_success']} for label in summaries]
    common=set.intersection(*successes)
    if common:
        seeds['all-success.html']=min(common)
    for name,replay_seed in seeds.items():
        replay=collect_replay(3,0.,100,5,replay_seed,models=models,
                              schedules=('adaptive','fixed','full'),terminate_on_collision=True,min_agent_distance=min_agent_distance)
        for run in replay['runs']:
            label='fixed_5' if run['schedule']=='fixed' else run['schedule']
            row=next(r for r in records if r['condition']==label and r['seed']==replay_seed)
            assert run['active_steps']==row['active_steps']
            assert run['frames'][-1]['attempts']==row['attempted_links']
            assert run['failed_by_collision']==row['any_collision']
            assert bool(run['frames'][-1]['success'])==row['final_success']
            assert abs(run['frames'][-1]['coverage']-row['final_coverage'])<1e-12
            assert abs(run['frames'][-1]['task_return']-row['task_return'])<1e-6
        serialized=json.dumps(replay,allow_nan=False).replace('<','\\u003c')
        (output/name).write_text(HTML.replace('__REPLAY_DATA__',serialized),encoding='utf-8')
    lines=['# Collision-terminated scheduler comparison','',f'Scope: {scope}.','',
           f'{episodes} matched scenarios per condition, seeds {seed}-{seed+episodes-1}; no packet loss.',
           f'Minimum pairwise agent starting center distance: {min_agent_distance} units, measured radially. Landmark spacing remains 0.4.',
           'Success requires collision-free survival through all 100 scheduled steps and full landmark coverage at the final step.',
           'Collisions present at reset fail immediately at step 0, with zero communication. Random starts are not filtered in evaluation.','',
           '| Scheduler | Success (95% Wilson interval) | Episodes with any collision | Attempts / 600 | Packets/episode | Attempts/active step | Mean steps to termination |',
           '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for label,m in summaries.items():
        lo,hi=m['success_95pct_wilson']
        active='undefined' if m['attempts_per_active_step'] is None else f"{m['attempts_per_active_step']:.3f}"
        lines.append(f"| {label} | {m['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {m['collision_episode_fraction']:.1%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {active} | {m['mean_active_steps']:.2f} |")
    lines+=['','## Termination details','',
            '| Scheduler | Initial-collision episodes | Survived to horizon | Median steps | Min-max steps | Mean steps among collision failures |',
            '| --- | --- | --- | ---: | --- | ---: |']
    for label,m in summaries.items():
        conditional='undefined' if m['mean_steps_before_collision'] is None else f"{m['mean_steps_before_collision']:.2f}"
        lines.append(f"| {label} | {m['initial_collision_fraction']:.1%} | {m['horizon_survival_fraction']:.1%} | {m['median_active_steps']:.1f} | {m['min_active_steps']}-{m['max_active_steps']} | {conditional} |")
    bound=next(iter(summaries.values()))['success_upper_bound_given_initial_collisions']
    lines+=['',f'Initial-collision starts impose a success ceiling of {bound:.1%} on this scenario set, even with perfect subsequent control.',
            'Surviving to step 100 without full coverage is a failure but not a collision episode.',
            'Attempts per active step is total attempted packets divided by total executed environment steps across all episodes. Zero-step failures contribute neither attempts nor active steps. The fixed 600 denominator includes all scheduled 100-step episodes, even early failures.',
            'For full communication, attempts per active step is exactly 6. Fixed_5 sends at active steps 1, 6, 11, ...; its aggregate attempts per active step can exceed 1.2 because episodes may end between sending steps.',
            'Shortened task returns are not directly comparable to previous 100-step returns. Each row uses its own trained policy; training loss and termination settings are recorded per checkpoint in summary.json. The scope above distinguishes frozen-policy testing from collision-aware retraining.',
            'Single training seed per condition. Wilson intervals measure episode uncertainty, not training-seed variation. Checkpoint configurations and SHA-256 hashes are recorded in summary.json.',
            '', '[First-scenario replay](comparison.html). Per-episode metrics are in episodes.jsonl.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'complete','output':str(output),'conditions':summaries}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/collision-zero-loss-20261104/frozen'))
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--seed',type=int,default=400000)
    parser.add_argument('--adaptive',type=Path,default=Path('results/adaptive-loss25-20261104-constrained/training/target.pt'))
    parser.add_argument('--fixed',type=Path,default=Path('results/reproduction-20261028/spaced-navigation/fixed_5/best.pt'))
    parser.add_argument('--full',type=Path,default=Path('results/reproduction-20261028/spaced-navigation/full/best.pt'))
    parser.add_argument('--scope',default='existing frozen policies; no retraining')
    parser.add_argument("--min-agent-distance",type=float,default=0.4)
    args=parser.parse_args()
    if args.episodes<1 or args.seed<0:
        parser.error('positive episode count and nonnegative seed required')
    evaluate_study({'adaptive':args.adaptive,'fixed_5':args.fixed,'full':args.full},args.output,args.episodes,args.seed,args.scope,min_agent_distance=args.min_agent_distance)


if __name__=='__main__':
    main()
