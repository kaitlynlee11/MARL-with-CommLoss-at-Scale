from types import SimpleNamespace

import numpy as np
import pytest

from limited_comm.env import CommunicationSpread
from limited_comm.rewards import TASK_REWARD_VERSION
from limited_comm.train import adaptive_reward
from limited_comm.vector_env import VectorSpread


@pytest.mark.parametrize("loss", [0.0, 1.0])
@pytest.mark.parametrize("comm_cost", [0.0, 0.12])
def test_covered_return_plateaus_but_packets_still_cost_and_lost_coverage_costs(loss, comm_cost):
    env = CommunicationSpread(horizon=10, loss=loss, comm_cost=comm_cost,
                              terminate_on_collision=True)
    vector = VectorSpread(batch=1, horizon=10, loss=loss, comm_cost=comm_cost)
    try:
        env.reset(seed=7)
        for i, (agent, landmark) in enumerate(zip(env.world.agents, env.world.landmarks)):
            landmark.state.p_pos[:] = [i - 1, 0]
            agent.state.p_pos[:] = [i - 1, 0.05]  # Covered, away from exact center.
            agent.state.p_vel[:] = 0
        vector.landmarks[0] = [l.state.p_pos for l in env.world.landmarks]
        vector.pos[0] = [a.state.p_pos for a in env.world.agents]
        vector.vel.fill(0)
        task_return = 0.0
        for _ in range(4):
            _, rewards, terminated, truncated, infos = env.step(
                {a: np.array([0, 1]) for a in env.agents})
            _, _, vreward, boundary, vinfo = vector.step(np.zeros((1, 3), int), np.ones((1, 3), int))
            info = infos['agent_0']
            task_return += info['task_reward']
            assert task_return == 0.0
            assert info['success'] and info['mean_landmark_distance'] > 0
            assert info['task_reward_version'] == TASK_REWARD_VERSION
            assert not any(terminated.values()) and not any(truncated.values()) and not boundary
            assert info['attempted_links'] == 6
            assert info['delivered_links'] == (6 if loss == 0 else 0)
            assert rewards['agent_0'] == pytest.approx(-2 * comm_cost)
            assert vreward[0] == pytest.approx(rewards['agent_0'])
            assert vinfo['task_reward'][0] == 0

        # Reaching coverage is not an absorbing success or a permanently frozen score.
        for agent in env.world.agents:
            agent.state.p_pos[1] = 0.2
        vector.pos[0] = [a.state.p_pos for a in env.world.agents]
        _, rewards, _, _, infos = env.step({a: np.array([0, 0]) for a in env.agents})
        _, _, vreward, _, vinfo = vector.step(np.zeros((1, 3), int), np.zeros((1, 3), int))
        assert not infos['agent_0']['success']
        assert infos['agent_0']['task_reward'] == pytest.approx(-0.3)
        assert vreward[0] == pytest.approx(rewards['agent_0'])
        assert vinfo['task_reward'][0] == pytest.approx(-0.3)
    finally:
        env.close()


def test_collision_still_penalizes_and_terminates_even_with_all_landmarks_covered():
    env = CommunicationSpread(terminate_on_collision=True, comm_cost=0)
    vector = VectorSpread(batch=1)
    try:
        env.reset(seed=7)
        for agent, position in zip(env.world.agents, [[0.08, 0], [0.32, 0], [0.8, 0]]):
            agent.state.p_pos[:] = position
            agent.state.p_vel[:] = 0
        for landmark, position in zip(env.world.landmarks, [[0, 0], [0.4, 0], [0.8, 0]]):
            landmark.state.p_pos[:] = position
        vector.pos[0] = [a.state.p_pos for a in env.world.agents]
        vector.landmarks[0] = [l.state.p_pos for l in env.world.landmarks]
        _, rewards, terminated, truncated, infos = env.step({a: np.array([0, 0]) for a in env.agents})
        _, _, vreward, _, vinfo = vector.step(np.zeros((1, 3), int), np.zeros((1, 3), int))
        info = infos['agent_0']
        assert info['coverage'] == 1 and info['collision_pairs'] == 1
        assert not info['success'] and info['terminated_by_collision']
        assert all(terminated.values()) and not any(truncated.values())
        assert info['task_reward'] == pytest.approx(-1 / 3)
        assert vreward[0] == pytest.approx(rewards['agent_0'])
        assert vinfo['task_reward'][0] == pytest.approx(-1 / 3)
    finally:
        env.close()


def test_adaptive_distance_shaping_stops_at_full_coverage_without_hiding_packet_or_collision_costs():
    args = SimpleNamespace(coverage_bonus=1, success_bonus=10, collision_penalty=5,
                           packet_penalty=.05, time_penalty=.01, distance_penalty=.5)
    info = dict(coverage=np.array([1., 1., 1., 2 / 3]), success=np.array([True, True, False, False]),
                collision_pairs=np.array([0, 0, 1, 0]), attempted_links=np.array([0, 6, 0, 0]),
                distance_sum=np.array([.2, .2, .2, .4]))
    np.testing.assert_allclose(adaptive_reward(info, args), [13, 12.7, -2.01, 1.79])
