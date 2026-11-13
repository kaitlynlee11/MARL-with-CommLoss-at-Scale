import numpy as np
import pytest
import torch

from limited_comm.env import CommunicationSpread
from limited_comm.evaluate import evaluate_episode
from limited_comm.layout import minimum_separation, spaced_positions
from limited_comm.mappo import MAPPO


@pytest.mark.parametrize("n", [5, 10, 20])
def test_dense_team_resets_reproduce_and_preserve_both_radial_rules(n):
    env = CommunicationSpread(n=n, layout_sampler="sequential")
    try:
        for seed in range(400000, 400005):
            obs, _ = env.reset(seed=seed)
            agents = np.array([a.state.p_pos.copy() for a in env.world.agents])
            landmarks = np.array([l.state.p_pos.copy() for l in env.world.landmarks])
            assert minimum_separation(agents) >= .4 and minimum_separation(landmarks) >= .4
            assert (np.abs(agents) <= 1).all() and (np.abs(landmarks) <= 1).all()
            assert env.diagnostics()["collision_pairs"] == 0
            assert len(obs) == n and all(o.shape == (28,) for o in obs.values())
            env.reset(seed=seed)
            np.testing.assert_array_equal(agents, [a.state.p_pos for a in env.world.agents])
            np.testing.assert_array_equal(landmarks, [l.state.p_pos for l in env.world.landmarks])
            env.step({a: np.array([0, 0]) for a in env.agents})
            np.testing.assert_array_equal(landmarks, [l.state.p_pos for l in env.world.landmarks])
    finally:
        env.close()


@pytest.mark.parametrize("n", [5, 10, 20])
def test_actor_transfer_and_packet_denominators_use_actual_team_size(n):
    torch.set_num_threads(1)
    model = MAPPO(hidden=8, recurrent=True, adaptive=True)
    with torch.no_grad():
        model.actor.head.weight.zero_()
        model.actor.head.bias.fill_(-100)
        model.actor.head.bias[5] = 100  # hold still and transmit
    for schedule in ("adaptive", "full", "fixed", "none"):
        row = evaluate_episode(model, 400000, schedule, n=n, horizon=6, layout_sampler="sequential")
        expected = n * (n - 1) * {"adaptive": 6, "full": 6, "fixed": 2, "none": 0}[schedule]
        assert row["attempted_links"] == row["delivered_links"] == expected
        assert row["packet_ratio_fixed_denominator"] == expected / (n * (n - 1) * 6)
        assert row["agents"] == row["landmarks"] == n
        assert row["initial_collision"] is False


def test_invalid_layout_sampler_is_rejected():
    with pytest.raises(ValueError, match="sampler"):
        spaced_positions(np.zeros((20, 2)), np.random.default_rng(0), .4, method="bad")
