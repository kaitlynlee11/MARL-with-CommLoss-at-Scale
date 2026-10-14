import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from limited_comm.env import CommunicationSpread
from limited_comm.experiment import run_episode


def test_parallel_api():
    env = CommunicationSpread(horizon=10)
    try:
        parallel_api_test(env, num_cycles=20)
    finally:
        env.close()


@pytest.mark.parametrize("n", [3, 8])
def test_observation_shape_is_fixed_and_hidden_peers_do_not_leak(n):
    env = CommunicationSpread(n=n, radius=0.1)
    try:
        env.reset(seed=10)
        env.world.agents[0].state.p_pos[:] = [0, 0]
        for agent in env.world.agents[1:]:
            agent.state.p_pos[:] = [5, 5]
        obs = env._observations()["agent_0"]
        assert obs.shape == (28,)
        assert not obs[4:13].any()  # peer slots and masks are all zero
        assert not obs[-6:].any()  # reset does not preload messages
    finally:
        env.close()


def test_reset_clears_inbox_and_delivery_is_previous_state():
    env = CommunicationSpread(n=3)
    try:
        env.reset(seed=7)
        payloads = np.asarray([np.concatenate([a.state.p_pos, a.state.p_vel]) for a in env.world.agents])
        obs, _, _, _, info = env.step({a: np.array([0, 1]) for a in env.agents})
        np.testing.assert_allclose(obs["agent_0"][-6:-2], payloads[1:].mean(axis=0))
        assert info["agent_0"]["attempted_links"] == 6
        reset_obs, _ = env.reset(seed=7)
        assert not reset_obs["agent_0"][-6:].any()
    finally:
        env.close()


def test_schedule_accounting_and_random_navigation_control():
    records = {s: run_episode(3, 0.0, s, 5, 10, 42, 0.01) for s in ["none", "full", "fixed"]}
    assert records["none"]["attempted_links"] == 0
    assert records["full"]["attempted_links"] == 60
    assert records["fixed"]["attempted_links"] == 12
    # Movement is paired and ignores inboxes; physical task return must be identical.
    assert len({r["task_return"] for r in records.values()}) == 1
    assert records["full"]["penalized_return"] < records["none"]["penalized_return"]


def test_success_and_collision_definitions():
    env = CommunicationSpread(n=3)
    try:
        env.reset(seed=2)
        for i, (agent, landmark) in enumerate(zip(env.world.agents, env.world.landmarks)):
            agent.state.p_pos[:] = [i, 0]
            landmark.state.p_pos[:] = [i, 0]
        assert env.diagnostics()["success"]
        assert env.diagnostics()["collision_pairs"] == 0
        env.world.agents[1].state.p_pos[:] = env.world.agents[0].state.p_pos
        assert not env.diagnostics()["success"]
        assert env.diagnostics()["collision_pairs"] == 1
    finally:
        env.close()
