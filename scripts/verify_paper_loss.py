"""Audit completed frozen-policy packet-loss evaluations and replay link draws."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from limited_comm.collision_study import collision_summary
from limited_comm.layout import minimum_separation
from limited_comm.rewards import TASK_REWARD_VERSION, task_reward


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = read(root / 'summary.json')
    status, manifest = read(root / 'status.json'), result['manifest']
    assert status['state'] == status['phase'] == 'complete'
    assert status['completed'] == status['total'] == 6 * manifest['episodes_per_method_per_cohort']
    assert manifest['task_reward_version'] == TASK_REWARD_VERSION
    source = Path(manifest['source'])
    assert sha256(source / 'selection.json') == manifest['source_selection_sha256']
    for name, expected in manifest['source_code_sha256'].items():
        assert sha256(Path(name)) == expected, name
    for method, provenance in result['checkpoints'].items():
        assert sha256(Path(provenance['checkpoint'])) == provenance['sha256']
        assert provenance['training_loss'] == 0 and provenance['frozen']
    count, replay_count, frames_checked, attempts, lost = 0, 0, 0, 0, 0
    pooled = {method: [] for method in result['checkpoints']}
    for cohort, seed in manifest['cohorts'].items():
        expected_seeds = list(range(seed, seed + manifest['episodes_per_method_per_cohort']))
        records = {}
        for method in pooled:
            rows = [json.loads(s) for s in (root / cohort / f'{method}.episodes.jsonl').read_text().splitlines()]
            old_path = source / cohort / f'{method}.episodes.jsonl'
            old_rows = [json.loads(s) for s in old_path.read_text().splitlines()][:len(rows)]
            assert sha256(old_path) == result['paired_changes'][cohort][method]['sha256']
            assert [r['seed'] for r in rows] == expected_seeds
            for old, row in zip(old_rows, rows):
                assert row['seed'] == old['seed'] and row['condition'] == method
                assert row['loss'] == manifest['loss'] and row['task_reward_version'] == TASK_REWARD_VERSION
                assert row['terminate_on_collision'] and not row['initial_collision']
                assert row['actual_initial_agent_separation'] == old['actual_initial_agent_separation'] >= .4
                assert row['actual_min_separation'] == old['actual_min_separation'] >= .4
                assert row['attempted_links'] % 2 == 0
                assert row['packet_ratio_fixed_denominator'] == row['attempted_links'] / 600
                assert row['attempts_per_active_step'] == row['attempted_links'] / row['active_steps']
                assert 0 <= row['delivered_links'] <= row['attempted_links'] <= 6 * row['active_steps'] <= 600
                assert row['lost_links'] == row['attempted_links'] - row['delivered_links']
                assert row['payload_bytes'] == 16 * row['attempted_links']
                assert row['terminated_by_collision'] == row['any_collision']
                assert not row['final_success'] or (row['active_steps'] == 100 and not row['any_collision'] and row['final_coverage'] == 1)
                count += 1
                attempts += row['attempted_links']
                lost += row['lost_links']
            calculated = collision_summary(rows)
            saved = result['cohorts'][cohort][method]
            assert all(saved[k] == value for k, value in calculated.items())
            assert saved['total_attempted_links'] == sum(r['attempted_links'] for r in rows)
            assert saved['total_lost_links'] == sum(r['lost_links'] for r in rows)
            assert read(root / cohort / 'summary.json')[method] == saved
            records[method] = {r['seed']: r for r in rows}
            pooled[method].extend(rows)
        for path in sorted((root / cohort).glob('*.html')):
            text = path.read_text(encoding='utf-8')
            replay = json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', text, re.S).group(1))
            assert replay['loss'] == manifest['loss'] and replay['task_reward_version'] == TASK_REWARD_VERSION
            for method, run in zip(pooled, replay['runs']):
                row = records[method][replay['seed']]
                frames = run['frames']
                assert len(frames) == row['active_steps'] + 1
                assert minimum_separation(np.asarray(frames[0]['agents'])) >= .4
                assert minimum_separation(np.asarray(frames[0]['landmarks'])) >= .4
                rng = np.random.default_rng(np.random.SeedSequence(replay['seed']).spawn(1)[0])
                cumulative_attempts = cumulative_deliveries = 0
                reward_return = 0.0
                for frame in frames[1:]:
                    draws = rng.random((3, 3))
                    expected = {(s, r) for s in frame['sent_senders'] for r in range(3)
                                if r != s and draws[s, r] >= manifest['loss']}
                    assert {tuple(link) for link in frame['links']} == expected
                    cumulative_attempts += 2 * len(frame['sent_senders'])
                    cumulative_deliveries += len(expected)
                    assert frame['attempts'] == cumulative_attempts
                    assert frame['delivered'] == cumulative_deliveries
                    reward_return += float(task_reward(frame['mean_landmark_distance'] * 3,
                        frame['collision_pairs'], frame['coverage'] == 1, 3))
                    assert abs(frame['task_return'] - reward_return) < 1e-9
                    frames_checked += 1
                assert cumulative_attempts == row['attempted_links']
                assert cumulative_deliveries == row['delivered_links']
                assert abs(reward_return - row['task_return']) < 1e-9
                assert frames[-1]['success'] == row['final_success']
                replay_count += 1
    pooled_summary = {}
    for method, rows in pooled.items():
        m = collision_summary(rows)
        m['realized_loss_fraction'] = sum(r['lost_links'] for r in rows) / sum(r['attempted_links'] for r in rows)
        pooled_summary[method] = m
    report = {'verified': True, 'episodes': count, 'replay_policy_runs': replay_count,
              'replay_action_frames_checked': frames_checked, 'checkpoints_and_source_hashes_unchanged': True,
              'all_initial_layouts_match_zero_loss': True, 'directed_replay_delivery_draws_match_seeded_loss': True,
              'packet_denominator': 600, 'attempted_packets': attempts, 'lost_packets': lost,
              'realized_loss_fraction': lost / attempts, 'pooled': pooled_summary}
    (root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
