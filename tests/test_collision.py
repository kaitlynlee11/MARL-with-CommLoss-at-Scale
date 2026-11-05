import numpy as np
import pytest

from limited_comm.env import CommunicationSpread
from limited_comm.evaluate import evaluate_episode
from limited_comm.collision_study import collision_summary
from limited_comm.visualize import collect_replay


class Stay:
    spec={'adaptive':False}
    def reset_memory(self):
        pass
    def moves(self,obs):
        return np.zeros(obs.shape[:-1],dtype=int)


def place(monkeypatch,collision=False):
    original=CommunicationSpread.reset
    def reset(self,*args,**kwargs):
        original(self,*args,**kwargs)
        positions=[[-.7,-.7],[.7,-.7],[0,.7]]
        if collision:
            positions[1]=[-.7,-.7]
        for agent,pos in zip(self.world.agents,positions):
            agent.state.p_pos[:]=pos
            agent.state.p_vel[:]=0
        for landmark,pos in zip(self.world.landmarks,[[-.7,-.7],[.7,-.7],[0,.7]]):
            landmark.state.p_pos[:]=pos
        return self._observations(),{a:{} for a in self.agents}
    monkeypatch.setattr(CommunicationSpread,'reset',reset)


def test_initial_collision_has_zero_steps_zero_packets_and_failure(monkeypatch):
    place(monkeypatch,True)
    row=evaluate_episode(Stay(),4,'full',terminate_on_collision=True,min_agent_distance=0)
    assert row['active_steps']==row['attempted_links']==0
    assert row['any_collision'] and not row['final_success']
    assert row['attempts_per_active_step'] is None
    replay=collect_replay(3,0,100,5,4,model=Stay(),schedules=('full',),terminate_on_collision=True,min_agent_distance=0)
    assert len(replay['runs'][0]['frames'])==1
    assert replay['runs'][0]['failed_by_collision']


@pytest.mark.parametrize('schedule,packets',[('full',600),('fixed',120)])
def test_collision_free_full_horizon_success_and_fixed_denominator(monkeypatch,schedule,packets):
    place(monkeypatch)
    row=evaluate_episode(Stay(),4,schedule,terminate_on_collision=True)
    assert row['active_steps']==100 and row['final_success']
    assert not row['any_collision']
    assert row['attempted_links']==packets
    assert row['packet_ratio_fixed_denominator']==packets/600
    assert row['attempts_per_active_step']==packets/100
    summary=collision_summary([row])
    assert summary['attempts_per_active_step']==packets/100


def test_collision_is_true_terminal_not_time_limit(monkeypatch):
    place(monkeypatch)
    env=CommunicationSpread(terminate_on_collision=True)
    try:
        env.reset(seed=1)
        env.world.agents[1].state.p_pos[:]=env.world.agents[0].state.p_pos + np.array([.05,0])
        _,_,term,trunc,infos=env.step({a:np.array([0,1]) for a in env.agents})
        assert all(term.values()) and not any(trunc.values())
        assert not env.agents and not infos[env.possible_agents[0]]['success']
        with pytest.raises(RuntimeError):
            env.step({})
    finally:
        env.close()


def test_vector_terminal_flags_and_partial_reset_preserve_other_worlds():
    from limited_comm.collision_vector_env import CollisionVectorSpread
    env=CollisionVectorSpread(batch=2,horizon=100,loss=0)
    env.pos[0]=np.array([[0,0],[.1,0],[.8,.8]])
    env.pos[1]=np.array([[-.7,-.7],[.7,-.7],[0,.7]])
    env.vel.fill(0)
    _,_,_,boundary,info=env.step(np.zeros((2,3),int),np.ones((2,3),int))
    np.testing.assert_array_equal(boundary,[True,False])
    np.testing.assert_array_equal(info['collision_terminal'],[True,False])
    before=env.pos[1].copy()
    env.reset_indices(boundary)
    np.testing.assert_array_equal(env.pos[1],before)
    np.testing.assert_array_equal(env.steps,[0,1])
    assert not env.diagnostics()['collision_pairs'].any()


def test_collision_validation_keeps_initial_failures_and_zero_packet_attempts(monkeypatch):
    from limited_comm.train import validate
    from limited_comm.vector_env import VectorSpread
    original=VectorSpread.reset
    def reset(self):
        original(self)
        self.pos[:]=np.array([[0,0],[.1,0],[.8,.8]])
        return self.observe(),self.state()
    monkeypatch.setattr(VectorSpread,'reset',reset)
    result=validate(Stay(),episodes=4,schedule='full',terminate_on_collision=True)
    assert result['success']==0 and result['collision_episode_fraction']==1
    assert result['mean_active_steps']==0 and result['attempted_packets']==0
