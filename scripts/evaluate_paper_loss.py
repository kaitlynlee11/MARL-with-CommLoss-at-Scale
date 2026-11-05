"""Test the three selected frozen adaptive policies under packet loss."""
import argparse
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

import torch

from limited_comm.collision_study import collision_summary
from limited_comm.evaluate import evaluate_episode
from limited_comm.mappo import load_policy
from limited_comm.rewards import TASK_REWARD_VERSION
from limited_comm.visualize import HTML, collect_replay


NAMES = {'baseline': 'Initial adaptive checkpoint',
         'control': 'Equal-budget joint PPO continuation',
         'proposed': 'Paper-inspired modular constrained PPO'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def progress(root, state):
    write_json(root / 'status.json', state)
    refresh = '<meta http-equiv="refresh" content="5">' if state['state'] == 'running' else ''
    links = ''.join(f'<li><a href="{p.relative_to(root).as_posix()}">{html.escape(p.relative_to(root).as_posix())}</a></li>'
                    for p in sorted(root.rglob('*.html')) if p.name != 'progress.html')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">' + refresh +
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Packet-loss evaluation</title>'
            '<style>body{max-width:960px;margin:36px auto;padding:0 20px;font:16px system-ui;background:#101827;color:#edf2fa}'
            'progress{width:100%;height:28px;accent-color:#55d6be}a{color:#7dd3fc}</style>'
            f'<h1>Frozen adaptive policies: {state["loss"]:.0%} packet loss</h1>'
            f'<p>{html.escape(state["state"])} / {html.escape(state["phase"])}</p>'
            f'<progress value="{state["completed"]}" max="{state["total"]}"></progress>'
            f'<p>{state["completed"]:,} / {state["total"]:,} evaluation episodes</p>'
            '<p>Three frozen checkpoints; no retraining. Collision termination, final full coverage, '
            '0.4-unit starting separation, and 600 scheduled packets per episode.</p>'
            '<p><a href="report.md">Results report</a></p><ul>' + links + '</ul></html>')
    (root / 'progress.html').write_text(page, encoding='utf-8')


def make_replay(directory, name, seed, loss, models, rows_by_method):
    replays = []
    for method, model in models.items():
        replay = collect_replay(3, loss, 100, 5, seed, model=model,
                                min_agent_distance=.4, min_landmark_distance=.4,
                                terminate_on_collision=True)
        run = replay['runs'][0]
        run['label'] = NAMES[method]
        run['description'] = f'Frozen checkpoint under {loss:.0%} independent directed packet loss.'
        row = next(r for r in rows_by_method[method] if r['seed'] == seed)
        final = run['frames'][-1]
        assert run['active_steps'] == row['active_steps']
        assert run['failed_by_collision'] == row['any_collision']
        assert final['success'] == row['final_success']
        assert final['coverage'] == row['final_coverage']
        assert final['attempts'] == row['attempted_links']
        assert final['delivered'] == row['delivered_links']
        assert abs(final['task_return'] - row['task_return']) < 1e-9
        replays.append(replay)
    assert all(r['runs'][0]['frames'][0]['agents'] == replays[0]['runs'][0]['frames'][0]['agents']
               and r['runs'][0]['frames'][0]['landmarks'] == replays[0]['runs'][0]['frames'][0]['landmarks']
               for r in replays)
    combined = {**replays[0], 'separately_trained': True,
                'runs': [r['runs'][0] for r in replays], 'bounds': max(r['bounds'] for r in replays)}
    combined['selection_note'] = ('First saved scenario in the cohort.' if name == 'comparison.html' else
                                  'First scenario successful for all three policies; selected for illustration only.')
    (directory / name).write_text(HTML.replace('__REPLAY_DATA__', json.dumps(combined, allow_nan=False).replace('<', '\\u003c')),
                                  encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/paper-guided-adaptive-20261104'))
    parser.add_argument('--output', type=Path, default=Path('results/paper-guided-loss10-20261105'))
    parser.add_argument('--loss', type=float, default=.1)
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.loss <= 1 or not 1 <= args.episodes <= 200:
        parser.error('loss must be in [0, 1]; episodes must be in [1, 200] to match the saved reference cohorts')
    source, root = args.source.resolve(), args.output.resolve()
    selection = json.loads((source / 'selection.json').read_text(encoding='utf-8'))
    reference = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    models, provenance = {}, {}
    for method in NAMES:
        path = source / 'selected' / f'{method}.pt'
        assert digest(path) == selection[method]['sha256']
        model, checkpoint = load_policy(path)
        cfg = checkpoint['config']
        assert model.spec['adaptive'] and cfg['loss'] == 0
        assert cfg['min_agent_distance'] == cfg['min_landmark_distance'] == .4
        assert cfg['radius'] == .6 and cfg['horizon'] == 100 and cfg['terminate_on_collision']
        models[method] = model
        provenance[method] = {**selection[method], 'checkpoint': str(path),
                              'training_loss': cfg['loss'], 'frozen': True}
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'scope': 'Frozen-policy robustness evaluation; no retraining or checkpoint reselection',
                'source': str(source), 'source_selection_sha256': digest(source / 'selection.json'),
                'loss': args.loss, 'loss_model': 'Independent Bernoulli loss per attempted directed packet',
                'loss_pairing': 'Identical seed-indexed draws for every directed link and step across policies, including silent senders',
                'actor_loss_observation': 'No loss-rate input or acknowledgments; original local observation and memory only',
                'episodes_per_method_per_cohort': args.episodes,
                'cohorts': {'matched': 400000, 'fresh': 600000},
                'fresh_note': 'Fresh relative to training/selection; these physical seeds were already tested at zero loss',
                'environment': {'n': 3, 'horizon': 100, 'radius': .6, 'coverage_radius': .1,
                                'min_agent_distance': .4, 'min_landmark_distance': .4,
                                'terminate_on_collision': True, 'fixed_packet_denominator': 600},
                'success_definition': 'Survive collision-free through step 100 and finish with simultaneous full landmark coverage',
                'task_reward_version': TASK_REWARD_VERSION,
                'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'mpe2', 'pettingzoo']},
                'source_code_sha256': {str(p): digest(p) for p in
                    [Path(__file__), *Path('src/limited_comm').glob('*.py')]}}
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'checkpoints.json', provenance)
    state = {'state': 'running', 'phase': 'initializing', 'loss': args.loss,
             'completed': 0, 'total': 6 * args.episodes}
    progress(root, state)
    started = time.monotonic()
    summaries, zero_loss, paired = {}, {}, {}
    try:
        for cohort, start_seed in manifest['cohorts'].items():
            directory = root / cohort
            directory.mkdir()
            summaries[cohort], zero_loss[cohort], paired[cohort], records = {}, {}, {}, {}
            for method, model in models.items():
                state['phase'] = f'{cohort}/{method}'
                progress(root, state)
                old_path = source / cohort / f'{method}.episodes.jsonl'
                old_rows = [json.loads(line) for line in old_path.read_text().splitlines()][:args.episodes]
                assert [r['seed'] for r in old_rows] == list(range(start_seed, start_seed + args.episodes))
                assert provenance[method]['sha256'] == reference['checkpoints'][method]['sha256']
                rows = []
                with (directory / f'{method}.episodes.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
                    for index, old in enumerate(old_rows):
                        row = {'condition': method, **evaluate_episode(model, start_seed + index, 'adaptive',
                               loss=args.loss, horizon=100, radius=.6, min_agent_distance=.4,
                               min_landmark_distance=.4, terminate_on_collision=True)}
                        assert not row['initial_collision'] and row['actual_initial_agent_separation'] >= .4
                        assert row['actual_initial_agent_separation'] == old['actual_initial_agent_separation']
                        assert row['actual_min_separation'] == old['actual_min_separation'] >= .4
                        assert 0 <= row['delivered_links'] <= row['attempted_links'] <= 6 * row['active_steps'] <= 600
                        assert row['payload_bytes'] == 16 * row['attempted_links']
                        assert row['packet_ratio_fixed_denominator'] == row['attempted_links'] / 600
                        assert row['attempts_per_active_step'] == row['attempted_links'] / row['active_steps']
                        assert not row['final_success'] or (row['active_steps'] == 100 and not row['any_collision'])
                        assert row['terminated_by_collision'] == row['any_collision']
                        row['lost_links'] = row['attempted_links'] - row['delivered_links']
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        rows.append(row)
                        state['completed'] += 1
                        if (index + 1) % 10 == 0 or index + 1 == args.episodes:
                            state['elapsed_seconds'] = time.monotonic() - started
                            progress(root, state)
                            print(f'{cohort}/{method}: {index + 1}/{args.episodes}; total {state["completed"]}/{state["total"]}', flush=True)
                records[method] = rows
                summary = collision_summary(rows)
                attempts = sum(r['attempted_links'] for r in rows)
                lost = sum(r['lost_links'] for r in rows)
                summary.update(total_attempted_links=attempts, total_lost_links=lost,
                               realized_loss_fraction=lost / attempts if attempts else None,
                               lost_links=lost / len(rows))
                summaries[cohort][method] = summary
                old_summary = collision_summary(old_rows)
                # Historical task return uses the old scoring and is deliberately excluded.
                zero_loss[cohort][method] = {k: old_summary[k] for k in
                    ['success_rate', 'collision_episode_fraction', 'attempted_packet_ratio',
                     'attempts_per_active_step', 'mean_active_steps']}
                paired[cohort][method] = {
                    'zero_loss_episode_log': str(old_path), 'sha256': digest(old_path),
                    'success_to_failure': sum(o['final_success'] and not r['final_success'] for o, r in zip(old_rows, rows)),
                    'failure_to_success': sum(not o['final_success'] and r['final_success'] for o, r in zip(old_rows, rows)),
                    'success_delta_percentage_points': 100 * (summary['success_rate'] - old_summary['success_rate']),
                    'collision_delta_percentage_points': 100 * (summary['collision_episode_fraction'] - old_summary['collision_episode_fraction'])}
                write_json(directory / 'summary.json', summaries[cohort])
            state['phase'] = f'{cohort}/replays'
            progress(root, state)
            make_replay(directory, 'comparison.html', start_seed, args.loss, models, records)
            common = set.intersection(*[{r['seed'] for r in records[m] if r['final_success']} for m in NAMES])
            if common:
                make_replay(directory, 'all-success.html', min(common), args.loss, models, records)
        for method in NAMES:
            assert digest(Path(provenance[method]['checkpoint'])) == provenance[method]['sha256']
        result = {'manifest': manifest, 'checkpoints': provenance, 'cohorts': summaries,
                  'zero_loss_reference': zero_loss, 'paired_changes': paired,
                  'verification': {'episodes_checked': state['completed'], 'checkpoints_unchanged': True,
                                   'initial_layouts_match_zero_loss': True, 'replay_endpoints_match_logs': True}}
        write_json(root / 'summary.json', result)
        lines = [f'# Frozen adaptive policies at {args.loss:.0%} packet loss', '',
                 f'{args.episodes} episodes per policy in each of two cohorts; {state["completed"]:,} total evaluations. '
                 'All three selected zero-loss checkpoints are frozen. No retraining, tuning or checkpoint reselection.', '',
                 'Independent loss applies to each directed attempted packet. Policies share the seed-indexed loss draws '
                 'and original physical starting layouts. All-to-all messages, sensing, delays, cache TTL and physics are unchanged. '
                 'Success requires collision-free survival through step 100 and final simultaneous full coverage. '
                 'Agent and landmark initial separation remain at least 0.4 units. Packet denominator stays 600 even after early failure.', '',
                 '| Cohort | Policy | Success (95% Wilson) | Any collision | Attempts/600 | Attempts/active step | Mean steps | Realized packet loss |',
                 '| --- | --- | --- | --- | --- | ---: | ---: | --- |']
        for cohort, methods in summaries.items():
            for method, m in methods.items():
                lo, hi = m['success_95pct_wilson']
                lines.append(f'| {cohort} | {NAMES[method]} | {m["success_rate"]:.1%} ({lo:.1%}-{hi:.1%}) | '
                             f'{m["collision_episode_fraction"]:.1%} | {m["attempted_packet_ratio"]:.2%} | '
                             f'{m["attempts_per_active_step"]:.3f} | {m["mean_active_steps"]:.2f} | {m["realized_loss_fraction"]:.2%} |')
        lines += ['', '## Change from the same scenarios at zero loss', '',
                  '| Cohort | Policy | Success: 0% to test loss | Success change (pp) | Collision: 0% to test loss | Collision change (pp) |',
                  '| --- | --- | --- | ---: | --- | ---: |']
        for cohort, methods in summaries.items():
            for method, m in methods.items():
                old, delta = zero_loss[cohort][method], paired[cohort][method]
                lines.append(f'| {cohort} | {NAMES[method]} | {old["success_rate"]:.1%} to {m["success_rate"]:.1%} | '
                             f'{delta["success_delta_percentage_points"]:+.1f} | '
                             f'{old["collision_episode_fraction"]:.1%} to {m["collision_episode_fraction"]:.1%} | '
                             f'{delta["collision_delta_percentage_points"]:+.1f} |')
        lines += ['', 'Matched seeds: 400000 onward; fresh cohort: 600000 onward. "Fresh" refers to the original '
                  'training/selection split; both cohorts were previously tested at zero loss. One seeded loss realization '
                  'per scenario is used. Wilson intervals describe episode uncertainty for these fixed checkpoints, '
                  'not variation across training seeds. Attempted packets remain charged even when dropped; realized loss '
                  'need not equal the configured probability exactly.', '',
                  f'Task return uses `{TASK_REWARD_VERSION}`: distance costs stop while full coverage is maintained, '
                  'and collision costs remain. Historical zero-loss task returns use earlier scoring and are not compared. '
                  'No new reward training was performed.', '',
                  '[Matched comparison](matched/comparison.html) | [Fresh comparison](fresh/comparison.html)',
                  'Additional `all-success.html` replays, when present, select the first common successful scenario '
                  'for illustration; aggregate statistics use every episode.', '',
                  'Per-episode JSONL files, checkpoint hashes/configurations, source hashes, paired outcome changes, '
                  'realized drop counts and complete summaries are saved alongside this report.', '']
        (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        state.update(state='complete', phase='complete', elapsed_seconds=time.monotonic() - started)
        progress(root, state)
        print(json.dumps({'output': str(root), 'episodes': state['completed'], 'cohorts': summaries}, indent=2), flush=True)
    except BaseException:
        state.update(state='failed', phase='failed', error=traceback.format_exc())
        progress(root, state)
        raise


if __name__ == '__main__':
    main()
