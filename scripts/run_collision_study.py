"""Evaluate frozen policies, retrain collision-aware schedulers, then compare."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from limited_comm.collision_study import evaluate_study
from limited_comm.mappo import load_policy


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def latest(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2)
            stream.seek(max(0,stream.tell()-16000))
            lines=stream.read().decode('utf-8').splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                pass
    except OSError:
        pass
    return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/collision-zero-loss-20261104'))
    parser.add_argument('--initial-steps',type=int,default=8320000)
    parser.add_argument('--refine-steps',type=int,default=640000)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        args.initial_steps=args.refine_steps=6400
        args.episodes=3
    if min(args.initial_steps,args.refine_steps,args.episodes)<1 or args.initial_steps%6400 or args.refine_steps%6400:
        parser.error('positive budgets divisible by 6400 and a positive episode count required')
    repo=Path(__file__).resolve().parents[1]
    root=(repo/args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'manifest.json').open('x',encoding='utf-8') as stream:
        json.dump({**vars(args),'output':str(root),'training_seed':42,'test_seed':400000,'loss':0,
                   'collision_terminal':True,'min_agent_distance':.4,'fixed_denominator':600,'failure_penalty':100},stream,indent=2)
    stage={'state':'running','phase':'frozen-evaluation','pid':os.getpid()}
    workers={name:{'phase':'queued'} for name in ['adaptive','fixed_5','full']}
    finished=threading.Event()

    def monitor():
        while True:
            lines=[]
            details={}
            for label in workers:
                directory=root/'training'/label
                initial=latest(directory/'initial/learning.jsonl')
                refine=latest(directory/'refine/learning.jsonl')
                current=refine or initial
                steps=current.get('steps',0)
                budget=args.initial_steps+args.refine_steps
                metrics=current.get('validation') or read_json(directory/workers[label]['phase']/'status.json').get('validation',{})
                details[label]={**workers[label],'steps':steps,'budget':budget,'metrics':metrics}
                lines.append(f'<h2>{label}: {html.escape(workers[label]["phase"])}</h2><progress value="{steps}" max="{budget}"></progress><p>{steps:,} / {budget:,} training steps</p>')
                if metrics:
                    lines.append(f'<p>Validation collision-free success: {metrics["success"]:.1%}; collision episodes: {metrics.get("collision_episode_fraction",0):.1%}; packets/600: {metrics["packet_ratio"]:.2%}; active steps: {metrics.get("mean_active_steps",0):.2f}</p>')
            refresh='' if finished.is_set() else '<meta http-equiv="refresh" content="5">'
            links=''.join(f'<li><a href="{p.relative_to(root).as_posix()}">{html.escape(p.relative_to(root).as_posix())}</a></li>'
                          for p in sorted(root.rglob('*.html')) if p.name!='progress.html')
            page=('<!doctype html><html lang="en"><meta charset="utf-8">'+refresh+
                  '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Collision study progress</title>'
                  '<style>body{max-width:950px;margin:40px auto;padding:0 24px;background:#101827;color:#ecf2fa;font:16px system-ui}'
                  'progress{width:100%;height:24px;accent-color:#55d6be}a{color:#7dd3fc}</style>'
                  '<h1>Collision-terminated scheduler comparison: zero loss</h1><p>'+html.escape(stage['phase'])+' / '+html.escape(stage['state'])+
                  '</p><p>Updated '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'</p>'+''.join(lines)+
                  '<p>All starts have at least 0.4-unit pairwise radial agent separation. Budgets measure training work, not target attainment.</p><ul>'+links+'</ul></html>')
            (root/'progress.html').write_text(page,encoding='utf-8')
            (root/'status.json').write_text(json.dumps({**stage,'workers':details},indent=2),encoding='utf-8')
            if finished.wait(5):
                if stage.get('rendered'):
                    break
                stage['rendered']=True
    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()

    def child(label,phase,command):
        directory=root/'training'/label
        workers[label]['phase']=phase
        with (directory/f'{phase}.log').open('x',encoding='utf-8') as stream:
            cmd=[sys.executable,'-u','-m','limited_comm.train',*map(str,command)]
            stream.write(json.dumps(cmd)+'\n')
            stream.flush()
            process=subprocess.Popen(cmd,cwd=repo,stdout=stream,stderr=subprocess.STDOUT)
            workers[label]['pid']=process.pid
            if process.wait()!=0:
                raise RuntimeError(f'{label}/{phase} failed: inspect {directory/phase}.log')

    def train(label,schedule):
        directory=root/'training'/label
        directory.mkdir(parents=True)
        common=['--communication-mode','legacy','--schedule',schedule,'--period','5','--loss','0','--terminate-on-collision',
                '--min-agent-distance','.4',
                '--seed','42','--threads','1','--recurrent','--hidden','64','--eval-every','50',
                '--validation-episodes','200','--no-early-stop','--coverage-bonus','1','--success-bonus','10',
                '--collision-penalty','5','--failure-penalty','100','--packet-penalty','.05',
                '--time-penalty','.01','--distance-penalty','.5','--target-success','.9']
        child(label,'initial',['--output',directory/'initial','--steps',args.initial_steps,
                              '--entropy','.002','--gradient-clipping','joint',*common])
        child(label,'refine',['--resume',directory/'initial/last.pt','--output',directory/'refine',
                             '--steps',args.initial_steps+args.refine_steps,'--lr','.0003',
                             '--sequence-length','100','--minibatch','6400','--epochs','3',
                             '--entropy','.001','--gradient-clipping','separate',*common])
        candidates=[]
        for phase in ['initial','refine']:
            path=directory/phase/'best.pt'
            _,checkpoint=load_policy(path)
            m=checkpoint['validation']
            candidates.append(((m['success'],-m['collision_episode_fraction'],m['coverage'],-m['packet_ratio']),path,checkpoint))
        _,selected,checkpoint=max(candidates,key=lambda item:item[0])
        destination=directory/'best.pt'
        shutil.copyfile(selected,destination)
        selection={'source':str(selected),'selected_steps':checkpoint['steps'],'validation':checkpoint['validation'],
                   'training_work_steps':args.initial_steps+args.refine_steps,
                   'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
        (directory/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        workers[label]['phase']='complete'
        return label,destination

    try:
        original_paths={'adaptive':repo/'results/adaptive-loss25-20261104-constrained/training/target.pt',
                        'fixed_5':repo/'results/reproduction-20261028/spaced-navigation/fixed_5/best.pt',
                        'full':repo/'results/reproduction-20261028/spaced-navigation/full/best.pt'}
        frozen=read_json(root/'frozen/summary.json')
        if not frozen:
            frozen=evaluate_study(original_paths,root/'frozen',episodes=args.episodes)
        assert frozen['episodes_per_condition']==args.episodes and frozen['loss']==0
        assert frozen.get('min_agent_distance',0)==.4, 'Frozen evaluation must use the new 0.4 agent-start spacing'
        stage['phase']='collision-aware-retraining'
        paths={}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(train,label,schedule) for label,schedule in [('adaptive','adaptive'),('fixed_5','fixed'),('full','full')]]
            for future in as_completed(futures):
                label,path=future.result()
                paths[label]=path
        stage['phase']='retrained-evaluation'
        retrained=evaluate_study(paths,root/'retrained',episodes=args.episodes,
                                scope='three independently retrained collision-aware policies; matched two-phase budgets; zero training/evaluation loss')
        (root/'summary.json').write_text(json.dumps({'frozen':frozen,'retrained':retrained},indent=2),encoding='utf-8')
        lines=['# Frozen versus collision-aware retrained schedulers','',
               'Zero packet loss; all agent centers start at least 0.4 units apart radially. Collision terminates an episode. Success requires collision-free survival through step 100 and final full landmark coverage.',
               f'{args.episodes} matched held-out episodes per condition, seeds 400000 onward. Each new policy was trained from scratch for {args.initial_steps+args.refine_steps:,} world steps with training seed 42.',
               'Both phases use the same coverage/navigation reward, per-attempt communication charge and finite extra collision failure penalty of 100. Adaptive transmission additionally uses the packet-budget dual penalty. Checkpoints are selected on validation, never held-out outcomes.','',
               '| Phase | Scheduler | Collision-free success | Any-collision episodes | Attempts / 600 | Packets/episode | Attempts/active step | Mean termination step |',
               '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
        for phase,study in [('Frozen',frozen),('Retrained',retrained)]:
            for label,m in study['conditions'].items():
                lines.append(f"| {phase} | {label} | {m['success_rate']:.1%} | {m['collision_episode_fraction']:.1%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {m['attempts_per_active_step']:.3f} | {m['mean_active_steps']:.2f} |")
        lines+=['','## Interpretation','',
                'The 0.4-unit agent-start constraint prevents initial collisions. All 200 scenarios per condition are retained; no post-hoc reset-collision filtering is performed. The former 85% ceiling from unrestricted initial overlaps no longer applies.',
                'During training, zero-step initial failures are counted and automatically reset because no action transition exists to optimize. All actionable steps, including fatal collision steps, enter PPO; collision terminals have no value bootstrap, while time-limit truncations bootstrap normally. Recurrent memory resets per world at either boundary.',
                'Training packet dual updates use attempted traffic per active step relative to six full-communication attempts per active step (18% proxy budget). Evaluation always uses attempts/600 per scheduled episode and separately reports attempts per active step. Early failures can reduce attempts/600 even under full communication; the active-step metric exposes that effect.',
                'Frozen policies have different training histories, including 25% loss for the adaptive initializer study. Retraining is a new zero-loss, collision-aware, from-scratch single-seed experiment. These results do not isolate communication alone or quantify training-seed uncertainty.',
                '', '[Frozen report](frozen/report.md) | [Retrained report](retrained/report.md)',
                '[Frozen replay](frozen/comparison.html) | [Retrained replay](retrained/comparison.html)',
                '', 'Per-episode records, Wilson intervals, conditional collision-failure steps, checkpoint hashes and configurations are in the subdirectory reports and summaries.','']
        (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        stage.update(state='complete',phase='complete')
    except BaseException:
        stage.update(state='failed',phase='failed',error=traceback.format_exc())
        raise
    finally:
        finished.set()
        watcher.join(timeout=15)


if __name__=='__main__':
    main()
