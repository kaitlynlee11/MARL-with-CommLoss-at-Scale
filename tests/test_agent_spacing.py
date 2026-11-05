import numpy as np
import pytest

from limited_comm.env import CommunicationSpread
from limited_comm.vector_env import VectorSpread
from limited_comm.collision_vector_env import CollisionVectorSpread
from limited_comm.layout import minimum_separation, spaced_agents


@pytest.mark.parametrize('n',[3,5,8])
def test_default_agent_starts_are_radially_spaced_and_seeded(n):
    env=CommunicationSpread(n=n)
    try:
        for seed in range(25):
            env.reset(seed=seed)
            positions=np.array([a.state.p_pos.copy() for a in env.world.agents])
            assert minimum_separation(positions)>=.4
            assert (np.abs(positions)<=1).all()
            assert env.diagnostics()['collision_pairs']==0
            env.reset(seed=seed)
            np.testing.assert_array_equal(positions,[a.state.p_pos for a in env.world.agents])
    finally:
        env.close()


def test_batched_full_and_partial_resets_preserve_spacing():
    for kind in [VectorSpread,CollisionVectorSpread]:
        env=kind(batch=512,n=3,seed=15)
        for _ in range(5):
            env.reset()
            assert (minimum_separation(env.pos)>=.4).all()
        if kind is CollisionVectorSpread:
            np.testing.assert_array_equal(env.steps,np.zeros(env.batch,dtype=int))
            mask=np.arange(env.batch)%2==0
            untouched=env.pos[~mask].copy()
            for _ in range(5):
                env.reset_indices(mask)
                assert (minimum_separation(env.pos)>=.4).all()
                np.testing.assert_array_equal(env.pos[~mask],untouched)


def test_agent_spacing_is_radial_not_per_axis_and_is_start_only():
    diagonal=np.array([[0,0],[.3,.3],[-.8,.8]])
    np.testing.assert_array_equal(spaced_agents(diagonal,np.random.default_rng(3)),diagonal)
    env=VectorSpread(batch=1)
    env.pos[0]=[[0,0],[.35,0],[.8,.8]]
    env.vel[0]=[[1,0],[-1,0],[0,0]]
    env.step(np.zeros((1,3),int),np.zeros((1,3),int))
    assert minimum_separation(env.pos)[0]<.4


@pytest.mark.parametrize('bad',[-.1,np.nan,np.inf])
def test_invalid_agent_spacing_is_rejected(bad):
    with pytest.raises(ValueError):
        CommunicationSpread(min_agent_distance=bad)
    with pytest.raises(ValueError):
        VectorSpread(min_agent_distance=bad)


def test_zero_spacing_keeps_historical_unconstrained_option():
    env=CommunicationSpread(min_agent_distance=0)
    try:
        starts=[]
        for seed in range(100):
            env.reset(seed=seed)
            starts.append(minimum_separation(np.array([a.state.p_pos for a in env.world.agents])))
        assert min(starts)<.3
    finally:
        env.close()
