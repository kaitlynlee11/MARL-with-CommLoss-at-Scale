"""Run adaptive MAPPO, monitor progress, and evaluate the frozen selected policy."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/adaptive-loss25-20261104-constrained'))
    parser.add_argument('--initialize-actor', type=Path,
                        default=Path('results/reproduction-20261028/spaced-navigation/full/best.pt'))
    parser.add_argument('--steps', type=int, default=100_000_000, help='Maximum additional world steps; target stops earlier')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=(repo/args.output).resolve()
    initializer=(repo/args.initialize_actor).resolve()
    if not initializer.is_file():
        parser.error(f'missing navigation initializer: {initializer}')
    root.mkdir(parents=True,exist_ok=False)
    stage={'state':'running', 'stage':'training', 'target_met':False}
    finished=threading.Event()

    def monitor():
        while True:
            status=read_json(root/'training/status.json')
            metrics=status.get('validation',{})
            steps=status.get('steps',0)
            # Budget progress is not target progress: success and traffic are shown separately.
            body=f'<p>World steps: {steps:,} / {args.steps:,} maximum</p><progress value="{steps}" max="{args.steps}"></progress>'
            if metrics:
                success=metrics['success']
                ratio=metrics['packet_ratio']
                body+=(f'<h2>Final landmark coverage success: {success:.1%}</h2>'
                       f'<progress value="{success}" max="1"></progress><p>Target: strictly above 90%.</p>'
                       f'<h2>Attempted packet ratio: {ratio:.2%}</h2>'
                       f'<progress value="{ratio}" max="1"></progress><p>Target: strictly below 20% (under 120 packets/episode).</p>'
                       f'<p>Collision step rate: {metrics["collision_step_rate"]:.2%}</p>')
            links=''.join(f'<li><a href="{path.relative_to(root).as_posix()}">{html.escape(path.relative_to(root).as_posix())}</a></li>'
                          for path in root.rglob('*.html') if path.name!='progress.html')
            refresh='' if finished.is_set() else '<meta http-equiv="refresh" content="5">'
            page=('<!doctype html><html lang="en"><meta charset="utf-8">'+refresh+
                  '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Adaptive MAPPO progress</title>'
                  '<style>body{max-width:850px;margin:40px auto;padding:0 20px;font:16px system-ui;background:#101827;color:#eef4ff}'
                  'progress{width:100%;height:24px;accent-color:#55d6be}a{color:#7dd3fc}</style>'
                  '<h1>Adaptive scheduler: 25% packet loss</h1><p>'+html.escape(stage['stage'])+' / '+html.escape(stage['state'])+
                  '</p><p>Updated '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'</p>'+body+'<ul>'+links+'</ul></html>')
            (root/'progress.html').write_text(page,encoding='utf-8')
            (root/'status.json').write_text(json.dumps({**stage,'training':status},indent=2),encoding='utf-8')
            if finished.wait(5):
                # Render terminal state on next iteration before returning.
                if stage.get('_rendered'):
                    break
                stage['_rendered']=True

    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()
    def run(name,module,*arguments):
        stage['stage']=name
        cmd=[sys.executable,'-u','-m',module,*map(str,arguments)]
        with (root/f'{name}.log').open('x',encoding='utf-8') as log:
            log.write(json.dumps(cmd)+'\n')
            log.flush()
            subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('training','limited_comm.train','--output',root/'training','--schedule','adaptive',
            '--loss','.25','--recurrent','--hidden','64','--initialize-actor',initializer,
            '--steps',args.steps,'--threads','2','--lr','.0001','--epochs','3','--entropy','.002',
            '--sequence-length','25','--minibatch','2048','--eval-every','25',
            '--validation-episodes','200','--confirmation-episodes','200','--target-success','.9',
            '--packet-ratio-target','.2','--coverage-bonus','1','--success-bonus','10',
            '--collision-penalty','5','--packet-penalty','.05','--time-penalty','.01','--distance-penalty','.5')
        train_status=read_json(root/'training/status.json')
        stage['target_met']=train_status.get('target_met',False)
        checkpoint=root/'training'/('target.pt' if stage['target_met'] else 'best.pt')
        if train_status.get('state')=='user_stop':
            stage.update(state='user_stop',stage='stopped')
            return
        run('heldout-evaluation','limited_comm.evaluate',checkpoint,'--trained-schedule-only',
            '--loss','.25','--episodes','200','--seed','400000','--output',root/'evaluation')
        run('replay','limited_comm.visualize','--checkpoint',checkpoint,'--loss','.25',
            '--seed','400000','--output',root/'comparison.html')
        evaluation=read_json(root/'evaluation/summary.json')
        metrics=evaluation['conditions']['adaptive']
        ratio=metrics['attempted_links']/600
        result={'training':train_status,'confirmation_targets_met':stage['target_met'],
                'heldout_success':metrics['success_rate'],'heldout_packet_ratio':ratio,
                'heldout_targets_met':metrics['success_rate']>.9 and ratio<.2,
                'scope':'warm-start joint movement/transmit policy; one seed; 25% training and evaluation loss',
                'evaluation':evaluation}
        (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        lo,hi=metrics['success_95pct_wilson']
        (root/'report.md').write_text(
            '# Adaptive scheduler results\n\n'
            f'Training stop reason: {train_status["state"]}. Confirmation targets met: {stage["target_met"]}.\n\n'
            f'Held-out final success: {metrics["success_rate"]:.1%} (95% Wilson {lo:.1%} to {hi:.1%}).\n'
            f'Attempted packets per episode: {metrics["attempted_links"]:.2f} ({ratio:.2%} of full communication).\n'
            f'Coverage: {metrics["final_coverage"]:.1%}; collision step rate: {metrics["collision_step_rate"]:.2%}.\n'
            f'Held-out targets met: {result["heldout_targets_met"]}.\n\n'
            'The stopping set uses original-environment seeds 500000-500199; final evaluation uses '
            'baseline-matched seeds 400000-400199, without updating the policy. Both have 25% independent link loss. '
            'Intervals describe episodes, not variation across training seeds.\n',encoding='utf-8')
        stage.update(state='complete',stage='complete')
    except BaseException:
        stage.update(state='failed',stage='failed',error=traceback.format_exc())
        raise
    finally:
        finished.set()
        watcher.join(timeout=15)


if __name__=='__main__':
    main()
