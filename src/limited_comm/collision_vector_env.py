"""Asynchronous collision-terminal training worlds with zero-step reset failures."""
import numpy as np

from .layout import spaced_landmarks, spaced_agents
from .vector_env import VectorSpread


class CollisionVectorSpread(VectorSpread):
    """Reset only finished worlds; omit actionless initial failures from PPO buffers.

    Evaluation never filters starts. During training an initially terminal world
    contributes no action transition, so it is counted and immediately reset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps=np.zeros(self.batch,dtype=np.int64)
        self.initial_failures=0
        self.reset_indices(np.ones(self.batch,dtype=bool))

    def reset(self):
        # VectorSpread.__init__ calls reset before the collision counters exist.
        if not hasattr(self, "initial_failures"):
            return super().reset()
        return self.reset_indices(np.ones(self.batch,dtype=bool))

    def reset_indices(self, mask):
        pending=np.asarray(mask,dtype=bool).copy()
        while pending.any():
            ids=np.flatnonzero(pending)
            count=len(ids)
            self.pos[ids]=self.rng.uniform(-1,1,(count,self.n,2))
            self.vel[ids]=0
            landmarks=self.rng.uniform(-1,1,(count,self.n,2))
            self.landmarks[ids]=spaced_landmarks(landmarks,self.rng,self.min_landmark_distance)
            self.pos[ids]=spaced_agents(self.pos[ids],self.rng,self.min_agent_distance)
            self.cache[ids]=0
            self.age[ids]=self.ttl+1
            if self.landmark_channel is not None:
                reset_mask=np.zeros(self.batch,dtype=bool)
                reset_mask[ids]=True
                self.landmark_channel.reset(reset_mask)
            self.steps[ids]=0
            failed=self.diagnostics()['collision_pairs']>0
            pending &= failed
            self.initial_failures+=int(pending.sum())
        return self.observe(),self.state()

    def step(self,moves,gates):
        obs,state,reward,truncation,info=super().step(moves,gates)
        terminal=info['collision_pairs']>0
        info['collision_terminal']=terminal
        info['time_limit']=np.asarray(truncation)&~terminal
        info['success']=info['success']&~terminal
        return obs,state,reward,terminal|truncation,info
