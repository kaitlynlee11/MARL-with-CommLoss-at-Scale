import numpy as np
import pytest
import torch

from limited_comm.mappo import MAPPO, gae, load_policy
from limited_comm.sequential import ModularMAPPO, proximity_cost, select_movement, selection_score
from limited_comm.paper_train import clipped_loss, clipped_value_loss
from limited_comm.evaluate import evaluate_episode
from limited_comm.visualize import collect_replay


def test_modular_warm_start_preserves_joint_law_and_greedy_actions():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    parent=MAPPO(hidden=16,recurrent=True,adaptive=True)
    model=ModularMAPPO.from_joint(parent)
    old_memory=new_memory=None
    for _ in range(6):
        obs=torch.randn(2,3,28)
        old,old_memory=parent.policy(obs,old_memory)
        new,new_memory=model.policy(obs,new_memory)
        torch.testing.assert_close(old.probs,new.probs,atol=1e-7,rtol=1e-5)
        gd,conditional,_=model.actor.components(obs,None)
        g=torch.ones(2,3,dtype=torch.long)
        md=select_movement(conditional,g)
        assert md.logits.shape==(2,3,5)
        np.testing.assert_array_equal(parent.actions(obs.numpy()),model.actions(obs.numpy()))
    state=torch.randn(10,36)
    with_clock=torch.cat([state,torch.rand(10,1)],dim=-1)
    torch.testing.assert_close(parent.value(state),model.value(with_clock))
    with pytest.raises(RuntimeError):
        model.actions(with_clock.numpy())


def test_scheduler_gradient_does_not_update_behavior_parameters():
    torch.set_num_threads(1)
    model=ModularMAPPO(hidden=16)
    obs=torch.randn(2,3,28)
    gd,_,_=model.actor.components(obs)
    loss=-gd.log_prob(torch.ones(2,3,dtype=torch.long)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.actor.scheduler.parameters())
    assert all(p.grad is None for p in model.actor.behavior.parameters())


def test_modular_checkpoint_original_env_and_replay(tmp_path):
    torch.set_num_threads(1)
    model=ModularMAPPO(hidden=16)
    path=tmp_path/'new.pt'
    torch.save({'model_spec':model.spec,'model':model.state_dict()},path)
    restored,_=load_policy(path)
    obs=np.zeros((3,28),np.float32)
    np.testing.assert_array_equal(model.actions(obs),restored.actions(obs))
    row=evaluate_episode(restored,4,'adaptive',horizon=5,terminate_on_collision=True)
    replay=collect_replay(3,0,5,5,4,model=restored,terminate_on_collision=True)
    assert replay['runs'][0]['active_steps']==row['active_steps']
    assert replay['runs'][0]['frames'][-1]['attempts']==row['attempted_links']


def test_finite_horizon_risk_returns_do_not_bootstrap_or_cross_reset():
    cost=np.array([[0.],[1.],[0.]],np.float32)
    value=np.zeros_like(cost)
    next_value=np.full_like(cost,.9)
    done=np.array([[False],[True],[True]])
    _,returns=gae(cost,value,next_value,done,terminals=done,gamma=1,lam=1)
    # A nonterminal next value must be the value estimate at that next state;
    # use consistent values for the exact Monte Carlo identity.
    _,exact=gae(cost,value,value,done,terminals=done,gamma=1,lam=1)
    np.testing.assert_allclose(exact[:,0],[1,1,0])
    assert returns[1,0]==1 and returns[2,0]==0


def test_proximity_penalty_is_soft_and_does_not_change_positions():
    positions=np.array([[[0,0],[.6,0],[0,.9]],[[0,0],[.4,0],[0,.9]],[[0,0],[.3,0],[0,.9]]],float)
    before=positions.copy()
    costs=proximity_cost(positions)
    np.testing.assert_allclose(costs,[0,1/9,1],atol=1e-6)
    np.testing.assert_array_equal(positions,before)


def test_clipped_surrogates_are_finite_and_penalize_large_value_changes():
    ratio=torch.tensor([1.,2.,.5],requires_grad=True)
    advantage=torch.tensor([1.,1.,-1.])
    loss=clipped_loss(ratio,advantage,.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(ratio.grad).all()
    assert clipped_value_loss(torch.tensor([1.]),torch.tensor([0.]),torch.tensor([1.]))>.1


def test_validation_selection_keeps_packet_budget_and_explicit_safety_tradeoff():
    safe=dict(success=.6,collision_episode_fraction=.05,packet_ratio=.1)
    risky=dict(success=.7,collision_episode_fraction=.3,packet_ratio=.1)
    over=dict(success=1.,collision_episode_fraction=0.,packet_ratio=.2)
    assert selection_score(safe)>selection_score(risky)
    assert selection_score(safe)>selection_score(over)
