"""Replay the first successful saved episode, verifying against the original log."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from limited_comm.layout import minimum_separation
from limited_comm.mappo import load_policy
from limited_comm.rewards import TASK_REWARD_VERSION
from limited_comm.visualize import HTML, collect_replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/paper-guided-adaptive-20261104'))
    parser.add_argument('--cohort',choices=['matched','fresh'],default='fresh')
    parser.add_argument('--method',choices=['baseline','control','proposed'],default='proposed')
    parser.add_argument('--output',type=Path,help='New replay filename; existing files are never overwritten')
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.root/args.cohort/f'{args.method}.episodes.jsonl').read_text().splitlines()]
    record=next((r for r in rows if r['final_success'] and not r['any_collision']),None)
    if record is None:
        raise RuntimeError('No successful episode exists in the selected saved test records')
    seed=record['seed']
    path=args.output or args.root/args.cohort/f'{args.method}-success-seed{seed}.html'
    if path.exists() or path.with_suffix('.json').exists():
        parser.error('output exists; choose a new --output filename')
    checkpoint=args.root/'selected'/f'{args.method}.pt'
    selection=json.loads((args.root/'selection.json').read_text())[args.method]
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==selection['sha256']
    torch.set_num_threads(1)
    model,_=load_policy(checkpoint)
    replay=collect_replay(3,record['loss'],100,record['period'],seed,model=model,
                         radius=.6,min_landmark_distance=.4,min_agent_distance=.4,terminate_on_collision=True)
    run=replay['runs'][0]
    final=run['frames'][-1]
    assert run['active_steps']==record['active_steps']==100
    assert final['success'] and final['coverage']==record['final_coverage']==1
    assert not run['failed_by_collision'] and all(f['collision_pairs']==0 for f in run['frames'])
    assert final['attempts']==record['attempted_links'] and final['delivered']==record['delivered_links']
    assert final['payload_bytes']==record['payload_bytes']
    # Historical logs used the center-distance cost even at full coverage.
    legacy_return=sum(-.5*3*f['mean_landmark_distance']-f['collision_pairs']/3 for f in run['frames'][1:])
    expected_return=final['task_return'] if record.get('task_reward_version')==TASK_REWARD_VERSION else legacy_return
    assert abs(expected_return-record['task_return'])<1e-6
    assert minimum_separation(np.asarray(run['frames'][0]['agents']))>=.4
    landmarks=np.asarray([frame['landmarks'] for frame in run['frames']])
    assert np.all(landmarks==landmarks[0]) and minimum_separation(landmarks[0])>=.4
    run['label']='Paper-inspired adaptive policy: successful fresh-test episode'
    run['description']=f'Seed {seed}; first successful saved fresh-test scenario selected for illustration.'
    replay['selection_note']='First saved success in this cohort; illustrative example, not a random or aggregate performance estimate.'
    replay['checkpoint_sha256']=digest
    replay['logged_episode']=record
    replay['return_recalculation']={'source_task_reward_version':record.get('task_reward_version','legacy-distance'),
                                  'source_task_return':record['task_return'],
                                  'note':'Task return uses the current reward rule; the original logged return is preserved separately. No retraining.'}
    content=HTML.replace('__REPLAY_DATA__',json.dumps(replay,allow_nan=False).replace('<','\\u003c'))
    with path.open('x',encoding='utf-8') as stream:
        stream.write(content)
    metadata={'source_record':record,'checkpoint':str(checkpoint.resolve()),'sha256':digest,
              'selection_rule':'first successful saved episode in selected cohort; no retraining or checkpoint selection',
              'task_reward_version':TASK_REWARD_VERSION,'task_return':final['task_return'],
              'replay_verified':True}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'replay':str(path.resolve()),'seed':seed,'steps':100,'final_coverage':final['coverage'],
                      'collision_pairs':final['total_collisions'],'attempted_packets':final['attempts'],
                      'packet_ratio':final['attempts']/600,'first_coverage_step':record['first_success_step']},indent=2))


if __name__=='__main__':
    main()
