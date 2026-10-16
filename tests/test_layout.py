import numpy as np
import pytest

from limited_comm.env import CommunicationSpread
from limited_comm.layout import minimum_separation, spaced_landmarks
from limited_comm.schedules import transmit


def test_spacing_is_radial_and_sampler_is_seeded():
    diagonal = np.array([[0, 0], [0.3, 0.3], [-0.8, 0.8]])
    np.testing.assert_array_equal(spaced_landmarks(diagonal, np.random.default_rng(1), 0.4), diagonal)
    initial = np.zeros((1000, 3, 2))
    a = spaced_landmarks(initial, np.random.default_rng(42), 0.4)
    b = spaced_landmarks(initial, np.random.default_rng(42), 0.4)
    np.testing.assert_array_equal(a, b)
    assert (minimum_separation(a) >= 0.4).all()
    assert (np.abs(a) <= 1).all()
    with pytest.raises(ValueError):
        spaced_landmarks(initial[:1], np.random.default_rng(42), 3)


def test_spacing_persists_all_steps_and_reset_reproduces():
    env = CommunicationSpread(horizon=10)
    try:
        env.reset(seed=18)
        positions = np.array([l.state.p_pos.copy() for l in env.world.landmarks])
        assert minimum_separation(positions) >= 0.4
        for _ in range(10):
            env.step({a: np.array([0, 1]) for a in env.agents})
            np.testing.assert_array_equal([l.state.p_pos for l in env.world.landmarks], positions)
        env.reset(seed=18)
        np.testing.assert_array_equal([l.state.p_pos for l in env.world.landmarks], positions)
    finally:
        env.close()


def test_schedule_phase_restarts_at_each_episode():
    for schedule, count in [("full", 100), ("fixed", 20), ("none", 0)]:
        assert sum(transmit(schedule, t, 5) for t in range(100)) == count
    assert [t for t in range(11) if transmit("fixed", t)] == [0, 5, 10]
    assert transmit("fixed", 0) == 1
