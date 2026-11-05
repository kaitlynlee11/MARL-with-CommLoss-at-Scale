"""Verify and document the full 0.4-agent-spacing collision experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from limited_comm.mappo import load_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/collision-zero-loss-agent-spacing04-20261104'))
    parser.add_argument('--report',type=Path,default=Path('docs/collision-zero-loss-agent-spacing04-results.md'))
    args=parser.parse_args()
    root=args.source.resolve()
    status=json.loads((root/'status.json').read_text())
    assert status['state']=='complete',status
    result=json.loads((root/'summary.json').read_text())
    reference_layouts={}
    verified=0
    for phase in ['frozen','retrained']:
        study=result[phase]
        assert study['min_agent_distance']==.4 and study['loss']==0 and study['fixed_packet_denominator']==600
        rows=[json.loads(line) for line in (root/phase/'episodes.jsonl').read_text().splitlines()]
        assert len(rows)==600
        verified+=len(rows)
        for label in ['adaptive','fixed_5','full']:
            subset=[row for row in rows if row['condition']==label]
            assert len(subset)==200 and {r['seed'] for r in subset}==set(range(400000,400200))
            metrics=study['conditions'][label]
            assert metrics['initial_collision_fraction']==0
            assert metrics['success_rate']==sum(r['final_success'] for r in subset)/200
            assert metrics['collision_episode_fraction']==sum(r['any_collision'] for r in subset)/200
            assert math.isclose(metrics['attempted_packet_ratio'],sum(r['attempted_links'] for r in subset)/120000)
            assert math.isclose(metrics['attempts_per_active_step'],sum(r['attempted_links'] for r in subset)/sum(r['active_steps'] for r in subset))
            for row in subset:
                assert row['min_agent_distance']==.4 and row['actual_initial_agent_separation']>=.4
                assert not row['initial_collision'] and 1<=row['active_steps']<=100
                assert row['attempted_links']==row['delivered_links']
                assert row['final_success']==(not row['any_collision'] and row['active_steps']==100 and row['final_coverage']==1)
                signature=(row['actual_initial_agent_separation'],row['actual_min_separation'])
                assert reference_layouts.setdefault(row['seed'],signature)==signature
                if label=='full':
                    assert row['attempted_links']==6*row['active_steps']
                elif label=='fixed_5':
                    assert row['attempted_links']==6*((row['active_steps']+4)//5)
            provenance=study['checkpoints'][label]
            assert hashlib.sha256(Path(provenance['checkpoint']).read_bytes()).hexdigest()==provenance['sha256']
            if phase=='retrained':
                config=provenance['training_config']
                assert config['min_agent_distance']==.4 and config['loss']==0 and config['terminate_on_collision']
                selection=json.loads((root/'training'/label/'selection.json').read_text())
                assert selection['training_work_steps']==8960000 and selection['sha256']==provenance['sha256']
                for part,steps in [('initial',8320000),('refine',8960000)]:
                    _,checkpoint=load_policy(root/'training'/label/part/'last.pt')
                    assert checkpoint['steps']==steps and checkpoint['config']['min_agent_distance']==.4
                    logs=[json.loads(line) for line in (root/'training'/label/part/'learning.jsonl').read_text().splitlines()]
                    assert all(r['initial_actionless_failures']==0 for r in logs)
                    assert all(math.isfinite(r[k]) for r in logs for k in ['mean_step_reward','policy_loss','value_loss','entropy','approx_kl'])
        for path in (root/phase).glob('*.html'):
            match=re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',path.read_text(encoding='utf-8'),re.S)
            replay=json.loads(match.group(1))
            assert replay['min_agent_distance']==.4 and replay['loss']==0 and replay['terminate_on_collision']
            for run in replay['runs']:
                label='fixed_5' if run['schedule']=='fixed' else run['schedule']
                row=next(r for r in rows if r['condition']==label and r['seed']==replay['seed'])
                final=run['frames'][-1]
                assert len(run['frames'])==row['active_steps']+1
                assert bool(final['success'])==row['final_success'] and final['attempts']==row['attempted_links']
                assert math.isclose(final['task_return'],row['task_return'],abs_tol=1e-6)
    report=args.report.resolve()
    assert not report.exists(),'Choose a new report filename.'
    def relative(path):
        return Path(os.path.relpath(path,report.parent)).as_posix()
    lines=['# Full collision-terminated experiments with 0.4-unit agent starts','',
           'Completed November 4, 2026. Both requested stages are complete: frozen-policy evaluation under the new reset distribution, followed by independent from-scratch retraining and reevaluation of adaptive, fixed-five-step and full-communication policies.','',
           '## Setup','',
           '- Three agents and three static landmarks; 100 scheduled steps; zero communication packet loss.',
           '- All agent centers start at least 0.4 units apart radially. Entire invalid agent layouts are resampled uniformly in the original [-1,1] square. This is a reset constraint; agents may move closer later.',
           '- Landmark center separation remains at least 0.4; local sensing radius remains 0.6; landmark coverage radius remains 0.1; overlapping agent centers closer than 0.3 constitute a collision.',
           '- Any collision immediately terminates the episode. Success requires collision-free survival through step 100 and full landmark coverage at that final step. Early full coverage does not end the episode.',
           '- Both stages use all 200 matched seeds 400000-400199 per scheduler. No scenarios are filtered or replaced after evaluation. There are no reset collisions and no initial-overlap-induced 85% ceiling.',
           '- Packet ratio is mean attempts divided by 600 packets per scheduled 100-step episode. Attempts per active step is total attempts divided by total executed steps. Later collision failures can still reduce the scheduled-episode packet ratio.','',
           '## Results','',
           '| Stage | Scheduler | Collision-free success (95% Wilson) | Any collision | Packets / 600 | Packets/episode | Attempts/active step | Mean termination step |',
           '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for phase in ['frozen','retrained']:
        for label,m in result[phase]['conditions'].items():
            lo,hi=m['success_95pct_wilson']
            lines.append(f"| {phase} | {label} | {m['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {m['collision_episode_fraction']:.1%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {m['attempts_per_active_step']:.3f} | {m['mean_active_steps']:.2f} |")
    lines+=['','## Termination details','',
            '| Stage | Scheduler | Horizon survival | Median active steps | Min-max active steps | Mean steps among collision failures |',
            '| --- | --- | ---: | ---: | --- | ---: |']
    for phase in ['frozen','retrained']:
        for label,m in result[phase]['conditions'].items():
            fatal='undefined' if m['mean_steps_before_collision'] is None else f"{m['mean_steps_before_collision']:.2f}"
            lines.append(f"| {phase} | {label} | {m['horizon_survival_fraction']:.1%} | {m['median_active_steps']:.1f} | {m['min_active_steps']}-{m['max_active_steps']} | {fatal} |")
    lines+=['','## Training and interpretation','',
            'Each new policy completed 8,320,000 initial plus 640,000 refinement world steps: 8,960,000 per policy and 26,880,000 total. All use training seed 42, a 64-unit recurrent actor and centralized critic, 64 parallel worlds, one CPU PyTorch thread per training process, and matched PPO budgets. Early stopping is disabled. Selection compares the best validation checkpoints from both phases; held-out outcomes are not used for selection.',
            'The initial phase uses LR 0.0005, entropy 0.002, five PPO epochs, 25-step sequences, minibatch 2048 and joint gradient clipping. Refinement uses LR 0.0003, entropy 0.001, three epochs, 100-step sequences, minibatch 6400 and separate clipping. Reset curriculum is disabled. Training and validation both enforce the new agent separation.',
            'The shared team reward is covered-landmark count + 10 times full coverage - 5 times collision pairs - per-attempt packet cost - 0.01 per unfinished step - 0.5 times nearest-landmark distance sum - an extra finite 100-point terminal collision penalty. The fixed packet charge is 0.05; adaptive transmission additionally uses a nonnegative dual multiplier updated toward an 18% stochastic active-step traffic proxy. This proxy is distinct from the reported attempts/600 metric.',
            'Collision terminals have zero value bootstrap; time-limit truncations bootstrap normally. Recurrent memory and physical state reset independently for finished worlds.',
            'Frozen policies retain the original pretrained weights: the adaptive target from the 25%-loss study and the earlier zero-loss fixed/full policies. They are tested here at zero loss with the new start distribution. The retrained policies are new from-scratch policies under the new reset and collision rules. These are different training histories, not a controlled isolation of communication alone.',
            'Previous safe-start filtered results excluded initial overlaps at the 0.3 collision threshold from an unrestricted-start experiment. The new 0.4 starting-distance distribution is stricter and resamples layouts; those prior 170-scenario results are not the same cohort as this new 200-scenario experiment.',
            'One training seed per condition. Wilson intervals describe episode uncertainty, not variation across training seeds. The experiment uses matched finite budgets; a completed run does not itself establish >90% success or the adaptive traffic target.','',
            '## Artifacts and verification','',
            f'- [Full run manifest]({relative(root/"manifest.json")})',
            f'- [Frozen and retrained summaries]({relative(root/"summary.json")})',
            f'- [Frozen report]({relative(root/"frozen/report.md")}) and [retrained report]({relative(root/"retrained/report.md")})',
            f'- [Frozen replay]({relative(root/"frozen/comparison.html")}) and [retrained replay]({relative(root/"retrained/comparison.html")})',
            f'- [Frozen episode records]({relative(root/"frozen/episodes.jsonl")}) and [retrained episode records]({relative(root/"retrained/episodes.jsonl")})',
            f'- [Finished progress page]({relative(root/"progress.html")})','',
            'The implementation passed 49 tests and an end-to-end spacing smoke pipeline before this full run. Final checks verified 1,200 comparative evaluation records; identical scenario seeds and starting-distance diagnostics across conditions/stages; every initial agent distance >=0.4; zero reset collisions; completed training budgets; finite optimization metrics; checkpoint hashes; packet accounting; and replay endpoints. Historical result files are preserved.','']
    text='\n'.join(lines)
    for target in re.findall(r'\]\(([^)]+)\)',text):
        assert (report.parent/target).resolve().exists(),target
    report.write_text(text,encoding='utf-8')
    (root/'verification.json').write_text(json.dumps({'verified_episodes':verified,'all_initial_agent_distances_at_least':.4,
        'reset_collisions':0,'matched_scenario_count':len(reference_layouts),'training_steps_per_policy':8960000,
        'checkpoint_hashes_verified':True,'replay_endpoints_verified':True},indent=2),encoding='utf-8')
    print(f'Verified {verified} evaluation episodes and all training budgets/hashes/replays; created {report}')
    for phase in ['frozen','retrained']:
        for label,m in result[phase]['conditions'].items():
            print(phase,label,{k:m[k] for k in ['success_rate','collision_episode_fraction','attempted_packet_ratio','attempted_links','attempts_per_active_step','mean_active_steps']})


if __name__=='__main__':
    main()
