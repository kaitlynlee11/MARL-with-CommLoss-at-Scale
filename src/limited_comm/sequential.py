"""Paper-inspired modular recurrent scheduling and behavior policies.

Binary scheduling, original local observations and original message semantics are
retained. This is not an implementation of multi-hop ACR-PPO or CS-MARL Top(q).
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .mappo import MAPPO, RecurrentActor, mlp


class ModularActor(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        # Two independent parameter sets allow actual alternating block updates.
        self.scheduler=RecurrentActor(inputs,hidden,10)
        self.behavior=RecurrentActor(inputs,hidden,10)
        self.hidden=hidden

    def _features(self, obs, memory=None):
        gm=None if memory is None else memory[...,0,:]
        bm=None if memory is None else memory[...,1,:]
        gate_logits,gm=self.scheduler(obs,gm)
        action_logits,bm=self.behavior(obs,bm)
        shape=obs.shape[:-1]
        gate_logits=torch.logsumexp(gate_logits.reshape(*shape,2,5),dim=-1)
        conditional=action_logits.reshape(*shape,2,5)
        return gate_logits,conditional,torch.stack([gm,bm],dim=-2)

    def components(self, obs, memory=None):
        logits,conditional,memory=self._features(obs,memory)
        return Categorical(logits=logits),conditional,memory

    def forward(self, obs, memory=None):
        gate_logits,conditional,memory=self._features(obs,memory)
        # Equal cloned heads cancel bit-for-bit at initialization, preserving the
        # teacher logits and greedy decisions, not merely a mathematical identity.
        offset=gate_logits-torch.logsumexp(conditional,dim=-1)
        joint=conditional+offset[...,None]
        return joint.flatten(-2),memory


class ModularMAPPO(MAPPO):
    def __init__(self,obs_dim=28,state_dim=37,hidden=64,recurrent=True,adaptive=True,separated=True):
        if not recurrent or not adaptive or not separated:
            raise ValueError('modular policy requires recurrent adaptive scheduling')
        super().__init__(obs_dim,state_dim,hidden,recurrent,adaptive)
        self.actor=ModularActor(obs_dim,hidden)
        self.packet_critic=mlp(state_dim,1,hidden)
        self.risk_critic=mlp(state_dim,1,hidden)
        for network,prior in [(self.packet_critic,.1),(self.risk_critic,.2)]:
            nn.init.zeros_(network[-1].weight)
            nn.init.constant_(network[-1].bias,np.log(prior/(1-prior)))
        self.spec['separated']=True

    def costs(self,state):
        return (self.packet_critic(state).squeeze(-1).sigmoid(),
                self.risk_critic(state).squeeze(-1).sigmoid())

    @classmethod
    def from_joint(cls,parent):
        if not parent.spec['adaptive'] or not parent.spec['recurrent'] or parent.spec.get('separated'):
            raise ValueError('initializer must be a joint recurrent adaptive checkpoint')
        model=cls(obs_dim=parent.spec['obs_dim'],state_dim=parent.spec['state_dim']+1,hidden=parent.spec['hidden'])
        model.actor.scheduler.load_state_dict(copy.deepcopy(parent.actor.state_dict()))
        model.actor.behavior.load_state_dict(copy.deepcopy(parent.actor.state_dict()))
        weights=copy.deepcopy(parent.critic.state_dict())
        original=weights['0.weight']
        weights['0.weight']=torch.cat([original,torch.zeros(original.shape[0],1)],dim=1)
        model.critic.load_state_dict(weights)
        for name in ['value_mean','value_std','value_updates']:
            getattr(model,name).copy_(getattr(parent,name))
        return model


def select_movement(conditional, gates):
    indices=gates[...,None,None].expand(*gates.shape,1,5)
    return Categorical(logits=conditional.gather(-2,indices).squeeze(-2))


def critic_state(env):
    """The clock is privileged critic input; actor observations stay length 28."""
    return np.concatenate([env.state(),(np.asarray(env.steps)/env.horizon)[:,None]],axis=-1).astype(np.float32)


def proximity_cost(positions,margin=.45,collision_distance=.3):
    """Smooth training-only penalty before contact; no action masking or forces."""
    if margin<=collision_distance:
        raise ValueError('safety margin must exceed collision distance')
    delta=positions[:,:,None,:]-positions[:,None,:,:]
    distance=np.linalg.norm(delta,axis=-1)
    penalty=np.clip((margin-distance)/(margin-collision_distance),0,1)**2
    return np.triu(penalty,k=1).sum(axis=(1,2)).astype(np.float32)


def constrained_advantage(reward_adv,packet_adv,risk_adv,packet_multiplier,risk_multiplier):
    combined=reward_adv-packet_multiplier*packet_adv-risk_multiplier*risk_adv
    return (combined-combined.mean())/(combined.std()+1e-8)


def selection_score(metrics):
    # Same rule is used by the equal-budget control. Safety is explicit without
    # silently labelling a safe, stationary, unsuccessful policy an improvement.
    return (int(metrics['packet_ratio']<.2),metrics['success']-2*metrics['collision_episode_fraction'],
            metrics['success'],-metrics['packet_ratio'])
