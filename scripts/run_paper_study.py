"""Controlled paper-guided adaptive-policy improvement with held-out tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import torch

from limited_comm.collision_study import collision_summary
from limited_comm.evaluate import evaluate_episode
from limited_comm.mappo import load_policy
from limited_comm.sequential import selection_score
from limited_comm.train import validate
from limited_comm.visualize import HTML, collect_replay


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/paper-guided-adaptive-20261104'))
    parser.add_argument('--source',type=Path,default=Path('results/collision-zero-loss-agent-spacing04-20261104/training/adaptive/best.pt'))
    parser.add_argument('--steps',type=int,default=3200000)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        args.steps=6400
        args.episodes=3
    if args.steps<6400 or args.steps%6400 or args.episodes<1:
        parser.error('steps must be positive multiples of 6400, episodes positive')
    repo=Path(__file__).resolve().parents[1]
    root=(repo/args.output).resolve()
    source=(repo/args.source).resolve()
    parent,checkpoint=load_policy(source)
    parent_steps=checkpoint['steps']
    root.mkdir(parents=True,exist_ok=False)
    (root/'selected').mkdir()
    torch.set_num_threads(1)
    initial=validate(parent,episodes=200,schedule='adaptive',loss=0,terminate_on_collision=True,min_agent_distance=.4)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    papers=[repo.parent/'23951_Scalable_Cooperative_Mul.pdf',repo.parent/'s10489-026-07202-4.pdf']
    manifest={**vars(args),'output':str(root),'source':str(source),'source_sha256':source_hash,'parent_steps':parent_steps,
              'additional_world_steps_per_arm':args.steps,'initial_validation':initial,'training_seed':42,
              'environment':{'loss':0,'horizon':100,'min_agent_distance':.4,'min_landmark_distance':.4,
                             'radius':.6,'terminate_on_collision':True,'fixed_packet_denominator':600},
              'selection':'packet ratio<.2; then success - 2*collision fraction; validation seed100000 only',
              'papers':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in papers]}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    phase={'state':'running','phase':'training','arms':{'control':{},'proposed':{}}}
    finished=threading.Event()
    def monitor():
        while True:
            panels=[]
            for arm in ['control','proposed']:
                state=read_json(root/arm/'status.json')
                total=state.get('steps',parent_steps if arm=='control' else 0)
                progress=max(0,total-parent_steps) if arm=='control' else total
                phase['arms'][arm]={'additional_steps':progress,'budget':args.steps,'training':state}
                panels.append(f'<h2>{arm}</h2><progress value="{progress}" max="{args.steps}"></progress><p>{progress:,} / {args.steps:,} additional steps</p>')
                m=state.get('validation',{})
                if m:
                    panels.append(f'<p>Validation: success {m["success"]:.1%}; collision episodes {m["collision_episode_fraction"]:.1%}; packets/600 {m["packet_ratio"]:.2%}</p>')
                if 'risk_multiplier' in state:
                    panels.append(f'<p>Risk multiplier: {state["risk_multiplier"]:.2f}; packet multiplier: {state["packet_multiplier"]:.2f}</p>')
            refresh='' if finished.is_set() else '<meta http-equiv="refresh" content="5">'
            links=''.join(f'<li><a href="{p.relative_to(root).as_posix()}">{html.escape(p.relative_to(root).as_posix())}</a></li>' for p in root.rglob('comparison.html'))
            page=('<!doctype html><html lang="en"><meta charset="utf-8">'+refresh+
                  '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper-guided policy optimization</title>'
                  '<style>body{max-width:900px;margin:40px auto;padding:0 20px;background:#101827;color:#edf2fa;font:16px system-ui}'
                  'progress{width:100%;height:25px;accent-color:#55d6be}a{color:#7dd3fc}</style>'
                  '<h1>Paper-guided adaptive optimization</h1><p>'+html.escape(phase['phase'])+' / '+html.escape(phase['state'])+'</p>'
                  '<p>Same zero-loss, collision-terminated environment; 0.4-unit agent/landmark starting separation.</p>'+''.join(panels)+'<ul>'+links+'</ul></html>')
            (root/'progress.html').write_text(page,encoding='utf-8')
            (root/'status.json').write_text(json.dumps(phase,indent=2),encoding='utf-8')
            if finished.wait(5):
                if phase.get('rendered'):
                    break
                phase['rendered']=True
    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()
    def train(arm):
        if arm=='proposed':
            command=['limited_comm.paper_train','--source',source,'--output',root/arm,'--steps',args.steps,'--threads','1']
        else:
            command=['limited_comm.train','--resume',source,'--output',root/arm,'--steps',parent_steps+args.steps,
                     '--schedule','adaptive','--loss','0','--terminate-on-collision','--min-agent-distance','.4',
                     '--recurrent','--hidden','64','--seed','42','--threads','1','--no-early-stop',
                     '--lr','.0001','--epochs','2','--sequence-length','25','--minibatch','6400','--entropy','.001',
                     '--gradient-clipping','separate','--eval-every','25','--validation-episodes','200',
                     '--coverage-bonus','1','--success-bonus','10','--collision-penalty','5','--failure-penalty','100',
                     '--packet-penalty','.05','--time-penalty','.01','--distance-penalty','.5','--target-success','.9',
                     '--reset-optimizer','--selection-collision-weight','2']
        cmd=[sys.executable,'-u','-m',*map(str,command)]
        with (root/f'{arm}.log').open('x',encoding='utf-8') as log:
            log.write(json.dumps(cmd)+'\n');log.flush()
            subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train,arm) for arm in ['control','proposed']]
            for future in futures:
                future.result()
        phase['phase']='selecting-checkpoints'
        proposed_initial=read_json(root/'proposed/initial-validation.json')
        assert all(abs(initial[key]-proposed_initial[key])<1e-10 for key in ['success','collision_episode_fraction','packet_ratio'])
        control_path=root/'control/best.pt'
        _,control=load_policy(control_path)
        if selection_score(initial)>selection_score(control['validation']):
            control_path=source
        selected={'baseline':source,'control':control_path,'proposed':root/'proposed/best.pt'}
        provenance={}
        for arm,path in selected.items():
            destination=root/'selected'/f'{arm}.pt'
            shutil.copyfile(path,destination)
            model,cp=load_policy(destination)
            provenance[arm]={'checkpoint':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                             'source_checkpoint':str(path),'selected_steps':cp['steps'],'validation':cp['validation'],
                             'model_spec':cp['model_spec'],'training_config':cp['config']}
        (root/'selection.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        summaries={}
        for cohort,seed in [('matched',400000),('fresh',600000)]:
            directory=root/cohort
            directory.mkdir()
            summaries[cohort]={}
            models={}
            for arm in ['baseline','control','proposed']:
                phase['phase']=f'{cohort}-evaluation/{arm}'
                model,_=load_policy(root/'selected'/f'{arm}.pt')
                models[arm]=model
                records=[]
                with (directory/f'{arm}.episodes.jsonl').open('x',encoding='utf-8',buffering=1) as log:
                    for index in range(args.episodes):
                        row={'condition':arm,**evaluate_episode(model,seed+index,'adaptive',loss=0,
                              min_landmark_distance=.4,min_agent_distance=.4,terminate_on_collision=True)}
                        assert not row['initial_collision'] and row['actual_initial_agent_separation']>=.4
                        assert row['attempted_links']==row['delivered_links']
                        records.append(row);log.write(json.dumps(row)+'\n')
                        if (index+1)%50==0:
                            print(f'{cohort}/{arm}: {index+1}/{args.episodes}',flush=True)
                summaries[cohort][arm]=collision_summary(records)
                (directory/'summary.json').write_text(json.dumps(summaries[cohort],indent=2),encoding='utf-8')
            replays=[]
            names={'baseline':'Initial adaptive checkpoint','control':'Equal-budget joint PPO continuation',
                   'proposed':'Modular constrained PPO (paper-inspired)'}
            for arm,model in models.items():
                replay=collect_replay(3,0,100,5,seed,model=model,min_agent_distance=.4,terminate_on_collision=True)
                run=replay['runs'][0]
                run['label']=names[arm];run['description']=names[arm]+'.'
                row=json.loads((directory/f'{arm}.episodes.jsonl').read_text().splitlines()[0])
                assert run['active_steps']==row['active_steps'] and run['frames'][-1]['attempts']==row['attempted_links']
                assert bool(run['frames'][-1]['success'])==row['final_success']
                replays.append(replay)
            combined={**replays[0],'separately_trained':True,'runs':[r['runs'][0] for r in replays],
                      'bounds':max(r['bounds'] for r in replays)}
            (directory/'comparison.html').write_text(HTML.replace('__REPLAY_DATA__',json.dumps(combined,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
        result={'manifest':manifest,'checkpoints':provenance,'cohorts':summaries}
        (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        lines=['# Paper-guided adaptive policy optimization','',
               'Same physical environment and evaluation contract: zero packet loss, three agents/landmarks, 0.4 radial starting separation, sensing radius 0.6, coverage radius 0.1, collision termination, final full-coverage success at step 100. Packet denominator stays 600 per scheduled episode.','',
               f'Both optimization arms start from the same selected adaptive checkpoint and receive {args.steps:,} additional world steps. The original baseline remains unchanged. The modular initial stochastic law and greedy validation match the original checkpoint. Both arms select checkpoints on the same 200 validation worlds using traffic<20%, then success - 2*collision fraction.',
               'The control continues joint PPO with fresh Adam, LR 0.0001, two epochs and 25-step recurrent sequences. The proposed method adds independent recurrent schedule/behavior blocks, alternating clipped updates with product importance correction, KL early-stop guards, separate reward/packet/collision critics, a critic-only clock, finite-horizon returns, projected packet/risk dual updates and a soft pre-collision penalty. Additional parameter count and optimizer work mean this is environment-step-matched, not FLOP-matched.','',
               '| Cohort | Policy | Success (95% Wilson) | Collision episodes | Attempts/600 | Packets/episode | Attempts/active step | Mean active steps |',
               '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
        for cohort,methods in summaries.items():
            for arm,m in methods.items():
                lo,hi=m['success_95pct_wilson']
                lines.append(f"| {cohort} | {arm} | {m['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {m['collision_episode_fraction']:.1%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {m['attempts_per_active_step']:.3f} | {m['mean_active_steps']:.2f} |")
        lines+=['','Matched cohort uses seeds 400000 onward for comparison with prior reports. Fresh cohort uses previously unused seeds 600000 onward. Neither cohort selects checkpoints or changes optimization settings. The risk budget is an empirical training target, not a collision guarantee. Point estimates and single-seed differences do not establish general superiority.',
                '', '[Matched replay](matched/comparison.html) | [Fresh replay](fresh/comparison.html)',
                '', 'Source PDFs, source hashes, optimizer settings, selected checkpoints and per-episode records are preserved in manifest.json, selection.json and the method/cohort directories.','']
        (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        phase.update(state='complete',phase='complete')
    except BaseException:
        phase.update(state='failed',phase='failed',error=traceback.format_exc())
        raise
    finally:
        finished.set();watcher.join(timeout=15)


if __name__=='__main__':
    main()
