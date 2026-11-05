"""Recalculate saved replay task returns, preserving original HTML and trajectories."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from limited_comm.rewards import TASK_REWARD_VERSION, task_reward
from limited_comm.visualize import HTML


def update(path):
    path = path.resolve(strict=True)
    original = path.read_bytes()
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                      original.decode('utf-8'), re.S)
    if match is None:
        raise ValueError(f'No embedded replay in {path}')
    replay = json.loads(match.group(1))
    if replay.get('task_reward_version') == TASK_REWARD_VERSION:
        print(f'Already current: {path}')
        return
    archive = path.with_name(path.stem + '.legacy-distance.html')
    metadata_path = path.with_suffix('.task-return-update.json')
    if archive.exists() or metadata_path.exists():
        raise FileExistsError('Refusing to overwrite an existing reward revision')
    endpoints = []
    n = replay['n']
    for run in replay['runs']:
        legacy = current = 0.0
        previous_return = run['frames'][0]['task_return']
        assert previous_return == 0
        for frame in run['frames'][1:]:
            distance = frame['mean_landmark_distance'] * n
            legacy += -.5 * distance - frame['collision_pairs'] / n
            if abs(legacy - frame['task_return']) > 1e-6:
                raise ValueError('Saved returns do not match the legacy reward rule')
            current += float(task_reward(distance, frame['collision_pairs'], frame['coverage'] == 1, n))
            if frame['coverage'] == 1 and frame['collision_pairs'] == 0:
                assert current == previous_return
            frame['task_return'] = current
            previous_return = current
        endpoints.append({'label': run.get('label', run['schedule']),
                          'legacy_task_return': legacy, 'task_return': current})
    revision = {'source_task_reward_version': 'legacy-distance',
                'task_reward_version': TASK_REWARD_VERSION,
                'source_sha256': hashlib.sha256(original).hexdigest(),
                'original_html': str(archive), 'endpoints': endpoints,
                'note': 'Only task return was recalculated. Original trajectories, packet counts, success and collision outcomes are unchanged; no retraining.'}
    replay['task_reward_version'] = TASK_REWARD_VERSION
    replay['return_recalculation'] = revision
    content = HTML.replace('__REPLAY_DATA__', json.dumps(replay, allow_nan=False).replace('<', '\\u003c'))
    with archive.open('xb') as stream:
        stream.write(original)
    with metadata_path.open('x', encoding='utf-8') as stream:
        json.dump(revision, stream, indent=2)
    path.write_text(content, encoding='utf-8')
    print(json.dumps({'path': str(path), **revision}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.paths:
        update(path)


if __name__ == '__main__':
    main()
