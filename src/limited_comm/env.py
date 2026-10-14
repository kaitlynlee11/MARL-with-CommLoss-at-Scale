"""PettingZoo ParallelEnv adapter around MPE2 Simple Spread physics.

World access is confined here because stock Simple Spread has silent agents.
Actors receive custom local observations, never the stock global observation.
"""
from functools import lru_cache

import numpy as np
from gymnasium import spaces
from mpe2 import simple_spread_v3
from pettingzoo import ParallelEnv

from .channel import Channel


class CommunicationSpread(ParallelEnv):
    metadata = {"name": "communication_spread_v0", "render_modes": []}

    def __init__(self, n=3, horizon=100, loss=0.0, comm_cost=0.01,
                 radius=0.6, neighbors=3, ttl=5, coverage_radius=0.1):
        if n < 2 or horizon < 1 or neighbors < 1 or radius <= 0 or coverage_radius <= 0:
            raise ValueError("Invalid environment dimensions or radii")
        if not 0 <= loss <= 1 or comm_cost < 0 or ttl < 1:
            raise ValueError("Invalid channel parameters")
        self.n, self.horizon, self.loss = n, horizon, loss
        self.comm_cost, self.radius, self.neighbors = comm_cost, radius, neighbors
        self.ttl, self.coverage_radius = ttl, coverage_radius
        self.base = simple_spread_v3.parallel_env(
            N=n, max_cycles=horizon, local_ratio=0.5, dynamic_rescaling=False)
        self.possible_agents = self.base.possible_agents[:]
        self.agents = []
        self.channel = None
        self.steps = 0

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Self pos/vel; K peers and K landmarks (dx, dy, mask); pooled inbox.
        return spaces.Box(-np.inf, np.inf, shape=(4 + 6 * self.neighbors + 6,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete([5, 2])  # movement, transmit gate

    @property
    def world(self):
        return self.base.unwrapped.world

    def reset(self, seed=None, options=None):
        self.base.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.steps = 0
        # Independent channel RNG avoids movement/action RNG coupling.
        self.channel = Channel(self.n, self.loss, self.ttl,
                               np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]))
        return self._observations(), {a: {} for a in self.agents}

    def _local_entities(self, origin, entities):
        relative = [entity.state.p_pos - origin for entity in entities]
        visible = sorted((p for p in relative if np.linalg.norm(p) <= self.radius),
                         key=lambda p: float(np.linalg.norm(p)))[:self.neighbors]
        slots = np.zeros((self.neighbors, 3), dtype=np.float32)
        for i, pos in enumerate(visible):
            slots[i] = [*pos, 1.0]
        return slots.ravel()

    def _observations(self):
        observations = {}
        for i, name in enumerate(self.possible_agents):
            agent = self.world.agents[i]
            observations[name] = np.concatenate([
                agent.state.p_pos, agent.state.p_vel,
                self._local_entities(agent.state.p_pos,
                                     [p for p in self.world.agents if p is not agent]),
                self._local_entities(agent.state.p_pos, self.world.landmarks),
                self.channel.observe(i)]).astype(np.float32)
        return observations

    def state(self):
        """Privileged centralized critic input; length depends on N."""
        return np.concatenate([*[np.concatenate([a.state.p_pos, a.state.p_vel])
                                 for a in self.world.agents],
                               *[l.state.p_pos for l in self.world.landmarks]]).astype(np.float32)

    def diagnostics(self):
        agents = self.world.agents
        distances = np.asarray([[np.linalg.norm(a.state.p_pos - l.state.p_pos)
                                 for a in agents] for l in self.world.landmarks])
        covered = distances.min(axis=1) <= self.coverage_radius
        collisions = sum(np.linalg.norm(a.state.p_pos - b.state.p_pos) < a.size + b.size
                         for i, a in enumerate(agents) for b in agents[i + 1:])
        return {"coverage": float(covered.mean()), "success": bool(covered.all()),
                "collision_pairs": int(collisions),
                "mean_landmark_distance": float(distances.min(axis=1).mean())}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("Reset before stepping, or episode has finished")
        active = self.agents[:]
        if set(actions) != set(active):
            raise ValueError("Provide exactly one action per active agent")
        if any(not self.action_space(a).contains(actions[a]) for a in active):
            raise ValueError("Actions must be [movement in 0..4, gate in 0..1]")
        gates = np.asarray([actions[a][1] for a in active])
        payloads = np.asarray([np.concatenate([a.state.p_pos, a.state.p_vel])
                               for a in self.world.agents], dtype=np.float32)
        traffic = self.channel.advance(gates, payloads)
        _, rewards, terminated, truncated, _ = self.base.step({a: int(actions[a][0]) for a in active})
        self.steps += 1
        observations = self._observations()
        diagnostics = self.diagnostics()
        # Charge attempted transmissions including packets lost in transit; shared team objective.
        penalty = self.comm_cost * traffic.attempted_links / self.n
        team_task_reward = float(np.mean(list(rewards.values())))
        rewards = {a: team_task_reward - penalty for a in active}
        info = {**diagnostics, "task_reward": team_task_reward,
                "communication_penalty": penalty, **vars(traffic)}
        infos = {a: info.copy() for a in active}
        self.agents = self.base.agents[:]
        return observations, rewards, terminated, truncated, infos

    def close(self):
        self.base.close()
