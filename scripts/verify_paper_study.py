"""Audit paper-guided experiment outputs and create the durable results report."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from limited_comm.mappo import load_policy
from limited_comm.sequential import selection_score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/paper-guided-adaptive-20261104'))
    parser.add_argument('--report',type=Path,default=Path('docs/paper-guided-optimization-results.md'))
    args=parser.parse_args()
    root=args.source.resolve()
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    result=json.loads((root/'summary.json').read_text())
    manifest=result['manifest']
    budget=manifest['additional_world_steps_per_arm']
    parent_steps=manifest['parent_steps']
    assert json.loads((root/'control/status.json').read_text())['steps']-parent_steps==budget
    assert json.loads((root/'proposed/status.json').read_text())['steps']==budget
    assert manifest['environment']['loss']==0 and manifest['environment']['min_agent_distance']==.4
    baseline=manifest['initial_validation']
    modular=json.loads((root/'proposed/initial-validation.json').read_text())
    assert all(baseline[k]==modular[k] for k in ['success','collision_episode_fraction','packet_ratio'])
    for paper in manifest['papers']:
        assert hashlib.sha256(Path(paper['path']).read_bytes()).hexdigest()==paper['sha256']
    for label,cp in result['checkpoints'].items():
        path=Path(cp['checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
        model,checkpoint=load_policy(path)
        assert model.spec['obs_dim']==28 and model.spec['adaptive']
        assert checkpoint['config']['loss']==0 and checkpoint['config']['min_agent_distance']==.4
        if label=='proposed':
            assert model.spec['state_dim']==37 and model.spec['separated']
    counts=0
    layouts={}
    for cohort,seed in [('matched',400000),('fresh',600000)]:
        all_rows={}
        for label,metrics in result['cohorts'][cohort].items():
            rows=[json.loads(line) for line in (root/cohort/f'{label}.episodes.jsonl').read_text().splitlines()]
            assert len(rows)==manifest['episodes']
            counts+=len(rows)
            all_rows[label]=rows
            assert {r['seed'] for r in rows}==set(range(seed,seed+len(rows)))
            assert math.isclose(metrics['success_rate'],sum(r['final_success'] for r in rows)/len(rows))
            assert math.isclose(metrics['collision_episode_fraction'],sum(r['any_collision'] for r in rows)/len(rows))
            assert math.isclose(metrics['attempted_packet_ratio'],sum(r['attempted_links'] for r in rows)/(600*len(rows)))
            for row in rows:
                assert not row['initial_collision'] and row['actual_initial_agent_separation']>=.4
                assert row['attempted_links']==row['delivered_links'] and 0<=row['attempted_links']<=6*row['active_steps']
                assert row['final_success']==(not row['any_collision'] and row['active_steps']==100 and row['final_coverage']==1)
                signature=(row['actual_initial_agent_separation'],row['actual_min_separation'])
                assert layouts.setdefault(row['seed'],signature)==signature
        text=(root/cohort/'comparison.html').read_text(encoding='utf-8')
        replay=json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',text,re.S).group(1))
        for label,run in zip(['baseline','control','proposed'],replay['runs']):
            row=all_rows[label][0]
            assert run['active_steps']==row['active_steps']
            assert run['frames'][-1]['attempts']==row['attempted_links']
            assert bool(run['frames'][-1]['success'])==row['final_success']
    report=args.report.resolve()
    assert not report.exists(),'Choose a new report path.'
    def link(path):
        return Path(os.path.relpath(path,report.parent)).as_posix()
    lines=['# Paper-guided adaptive optimization: measured results','',
           'Completed November 4, 2026. This is a controlled follow-up in the same zero-loss, collision-terminated, 0.4-agent-spacing environment. Both continuation arms received '+f'{budget:,} additional world steps from the identical existing adaptive checkpoint. The original checkpoint is retained as the baseline.','',
           '## Sources and implementation','',
           '[Method, equations, paper-to-code mapping and limitations](paper-guided-optimization.md). The supplied papers are [ACR-PPO](../../23951_Scalable_Cooperative_Mul.pdf), an anonymous ICLR 2026 review copy, and [CS-MARL](../../s10489-026-07202-4.pdf), Nama (2026), DOI 10.1007/s10489-026-07202-4. Neither paper supplies a safety guarantee for this implementation.',
           'The proposed method separates recurrent scheduling and behavior updates, uses clipped product importance ratios and KL early-stop guards, and adds separate reward, packet-cost and collision-risk critics. A clock is visible only to the critic. It uses finite-horizon returns, packet/risk dual ascent and a soft pre-collision training penalty. Message content, one-step delay, observations, sensing and physics are unchanged. This package adaptation is not an exact reproduction of either paper.',
           'The control uses the original joint PPO architecture and objective, but continues from the same checkpoint with a fresh optimizer, conservative LR 0.0001, two PPO epochs and 25-step recurrent sequences. The proposed method uses the same actor LR/epochs/sequence length, but a larger model and critic LR 0.0003. Equal environment steps are not equal optimizer FLOPs.',
           'Both arms use validation seed 100000 only for selection: packet ratio<20%, then success - 2*collision fraction, then success and lower traffic. The initial checkpoint is a candidate for both arms, and the modular initial law/greedy validation matches the original exactly. Neither held-out cohort selects checkpoints.','',
           '## Matched and fresh evaluation','',
           'Each cohort contains '+f"{manifest['episodes']} paired scenarios per policy. Matched uses seeds 400000 onward; fresh uses previously unused seeds 600000 onward. Success requires collision-free survival through step 100 and final full landmark coverage. Attempts/600 always uses the full scheduled episode, while attempts/active step uses executed steps. All starts satisfy the 0.4 radial agent-spacing rule; no episodes are filtered.",'',
           '| Cohort | Policy | Success (95% Wilson) | Collision episodes | Attempts / 600 | Packets/episode | Attempts/active step | Mean active steps |',
           '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for cohort,methods in result['cohorts'].items():
        for label,m in methods.items():
            lo,hi=m['success_95pct_wilson']
            lines.append(f"| {cohort} | {label} | {m['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {m['collision_episode_fraction']:.1%} | {m['attempted_packet_ratio']:.2%} | {m['attempted_links']:.2f} | {m['attempts_per_active_step']:.3f} | {m['mean_active_steps']:.2f} |")
    lines+=['','## Interpretation','']
    for cohort,methods in result['cohorts'].items():
        b,c,p=methods['baseline'],methods['control'],methods['proposed']
        lines.append(f"- {cohort}: proposed versus baseline changes success by {100*(p['success_rate']-b['success_rate']):+.1f} pp and collision episodes by {100*(p['collision_episode_fraction']-b['collision_episode_fraction']):+.1f} pp. Compared with the equal-budget control, changes are {100*(p['success_rate']-c['success_rate']):+.1f} pp success and {100*(p['collision_episode_fraction']-c['collision_episode_fraction']):+.1f} pp collisions.")
        strict=p['success_rate']>b['success_rate'] and p['collision_episode_fraction']<b['collision_episode_fraction'] and p['attempted_packet_ratio']<.2
        lines.append(f"  Simultaneously higher success, fewer collision episodes and traffic below 20% versus baseline: **{strict}** (point estimates).")
    fresh=result['cohorts']['fresh']['proposed']
    lines += ['',f"The proposed method's fresh-test collision fraction is {fresh['collision_episode_fraction']:.1%}, versus its 2% empirical training risk target. Its fresh success is {fresh['success_rate']:.1%}; the original >90% coverage target must be assessed separately from packet savings.",
              'Any collision reduction with lower coverage is a safety/coverage tradeoff, not an overall improvement. A result from one seed and 200 episodes is not a general algorithm superiority claim; Wilson intervals describe evaluation episodes only. Component ablations and independent training seeds are still needed.',
              'The risk multiplier has a finite cap and the KL guard stops future epochs after detecting excessive divergence; neither certifies collision avoidance. No action shield or hidden global-state actor input is used.','',
              '## Artifacts and checks','',
              f'- [Full numerical summary]({link(root/"summary.json")})',
              f'- [Manifest and PDF provenance]({link(root/"manifest.json")})',
              f'- [Selected checkpoint hashes and validation scores]({link(root/"selection.json")})',
              f'- [Matched replay]({link(root/"matched/comparison.html")}) and [fresh replay]({link(root/"fresh/comparison.html")})',
              f'- [Completed progress page]({link(root/"progress.html")})',
              '',f'Verified {counts} episode records, both additional-training budgets, unchanged PDF/source/checkpoint hashes, identical starting layouts across methods, zero initial collisions, packet accounting, exact initial validation equivalence and replay endpoints. The 56-test suite and complete two-arm smoke pipeline passed before full training. Historical checkpoints and reports were preserved.','']
    body='\n'.join(lines)
    for target in re.findall(r'\]\(([^)]+)\)',body):
        assert (report.parent/target).resolve().exists(),target
    report.write_text(body,encoding='utf-8')
    (root/'verification.json').write_text(json.dumps({'episode_records':counts,'budgets_verified':True,'hashes_verified':True,
        'initial_policy_equivalence':True,'replay_endpoints_verified':True,'initial_collisions':0},indent=2),encoding='utf-8')
    print(f'Verified {counts} episodes and created {report}')
    for cohort,methods in result['cohorts'].items():
        for label,m in methods.items():
            print(cohort,label,{k:m[k] for k in ['success_rate','collision_episode_fraction','attempted_packet_ratio','attempts_per_active_step','mean_active_steps']})


if __name__=='__main__':
    main()
