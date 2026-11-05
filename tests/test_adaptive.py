from types import SimpleNamespace

import numpy as np
import pytest
import torch

from limited_comm.mappo import MAPPO, load_policy
from limited_comm.train import adaptive_reward, meets_adaptive_target, update_packet_multiplier
from limited_comm.vector_env import VectorSpread
from limited_comm.evaluate import evaluate_episode
from limited_comm.visualize import collect_replay


def test_adaptive_action_encoding_memory_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    model = MAPPO(hidden=16, recurrent=True, adaptive=True)
    with torch.no_grad():
        model.actor.head.weight.zero_()
        model.actor.head.bias.fill_(-100)
        model.actor.head.bias[9] = 100
    obs = np.zeros((3, 28), dtype=np.float32)
    moves, gates = model.actions(obs)
    np.testing.assert_array_equal(moves, [4, 4, 4])
    np.testing.assert_array_equal(gates, [1, 1, 1])
    path = tmp_path / 'adaptive.pt'
    torch.save({'model_spec': model.spec, 'model': model.state_dict()}, path)
    restored, _ = load_policy(path)
    assert restored.action_dim == 10
    np.testing.assert_array_equal(restored.actions(obs), (moves, gates))
    # The original environment and viewer must execute the learned gates.
    row = evaluate_episode(restored, 12, 'adaptive', loss=1, horizon=4)
    assert row['attempted_links'] == 24
    assert row['delivered_links'] == 0
    replay = collect_replay(3, 1, 4, 5, 12, model=restored)
    assert replay['runs'][0]['schedule'] == 'adaptive'
    assert replay['runs'][0]['frames'][-1]['attempts'] == 24


def test_reward_charges_attempts_and_reserves_largest_bonus_for_full_coverage():
    args = SimpleNamespace(coverage_bonus=1, success_bonus=10, collision_penalty=5,
                           packet_penalty=.05, time_penalty=.01, distance_penalty=.5)
    info = dict(coverage=np.array([2/3, 1., 1., 1.]), success=np.array([False, True, True, True]),
                collision_pairs=np.array([0, 0, 1, 0]), attempted_links=np.array([0, 0, 0, 6]),
                distance_sum=np.zeros(4))
    reward = adaptive_reward(info, args)
    np.testing.assert_allclose(reward, [1.99, 13, 8, 12.7])


@pytest.mark.parametrize('success,ratio,expected', [(0.9, .1, False), (.91, .2, False),
                                                  (.91, .199, True), (.89, 0, False)])
def test_stop_requires_both_strict_targets(success, ratio, expected):
    args=SimpleNamespace(target_success=.9, packet_ratio_target=.2)
    assert meets_adaptive_target(dict(success=success, packet_ratio=ratio), args) == expected


def test_vector_attempts_include_failed_packets_and_silent_agents():
    env = VectorSpread(batch=2, loss=1, horizon=5)
    _, _, _, _, info = env.step(np.zeros((2, 3), dtype=int), np.array([[1,0,1],[0,0,0]]))
    np.testing.assert_array_equal(info['attempted_links'], [4,0])
    np.testing.assert_array_equal(info['delivered_links'], [0,0])


def test_joint_ppo_log_prob_replay_and_transmit_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model=MAPPO(hidden=16, recurrent=True, adaptive=True)
    obs=torch.randn(5,2,3,28)
    hidden=torch.zeros(2,3,16)
    actions, logps=[],[]
    with torch.no_grad():
        for x in obs:
            dist,hidden=model.policy(x,hidden)
            action=dist.sample()
            actions.append(action)
            logps.append(dist.log_prob(action))
    hidden=torch.zeros(2,3,16)
    losses=[]
    for t,x in enumerate(obs):
        dist,hidden=model.policy(x,hidden)
        torch.testing.assert_close(dist.log_prob(actions[t]), logps[t])
        losses.append(-dist.log_prob(actions[t]).mean())
    sum(losses).backward()
    assert model.actor.head.weight.grad[:5].abs().sum() > 0
    assert model.actor.head.weight.grad[5:].abs().sum() > 0


def test_dual_packet_penalty_increases_above_budget_and_projects_at_zero():
    assert update_packet_multiplier(0, .8, .02, .18) > 0
    assert update_packet_multiplier(0, .1, .02, .18) == 0
    assert update_packet_multiplier(.5, .1, .02, .18) < .5
    args = SimpleNamespace(coverage_bonus=1, success_bonus=10, collision_penalty=5,
                           packet_penalty=.05, time_penalty=.01, distance_penalty=.5)
    info = dict(coverage=np.array([1.]), success=np.array([True]), collision_pairs=np.array([0]),
                attempted_links=np.array([6]), distance_sum=np.array([0.]))
    assert adaptive_reward(info,args,packet_multiplier=.2)[0] == pytest.approx(11.5)
