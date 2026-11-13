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
from .landmark_channel import LandmarkChannel
from .layout import spaced_landmarks, spaced_agents
from .rewards import TASK_REWARD_VERSION, task_reward


class CommunicationSpread(ParallelEnv):
    metadata = {"name": "communication_spread_v0", "render_modes": []}

    def __init__(self, n=3, horizon=100, loss=0.0, comm_cost=0.01,
                 radius=0.6, neighbors=3, ttl=5, coverage_radius=0.1, min_landmark_distance=0.4, terminate_on_collision=False, min_agent_distance=0.4, layout_sampler="whole",
                 communication_mode="legacy", communication_radius=1.):
        if n < 2 or horizon < 1 or neighbors < 1 or radius <= 0 or coverage_radius <= 0:
            raise ValueError("Invalid environment dimensions or radii")
        if not 0 <= loss <= 1 or comm_cost < 0 or ttl < 1:
            raise ValueError("Invalid channel parameters")
        self.terminate_on_collision = terminate_on_collision
        self.n, self.horizon, self.loss = n, horizon, loss
        self.comm_cost, self.radius, self.neighbors = comm_cost, radius, neighbors
        self.ttl, self.coverage_radius = ttl, coverage_radius
        if not np.isfinite(min_landmark_distance) or min_landmark_distance < 0:
            raise ValueError("min_landmark_distance must be finite and nonnegative")
        if not np.isfinite(min_agent_distance) or min_agent_distance < 0:
            raise ValueError("min_agent_distance must be finite and nonnegative")
        self.min_agent_distance = min_agent_distance
        self.min_landmark_distance = min_landmark_distance
        if layout_sampler not in ("whole", "sequential"):
            raise ValueError("layout_sampler must be whole or sequential")
        self.layout_sampler = layout_sampler
        if communication_mode not in ("legacy", "landmark"):
            raise ValueError("communication_mode must be legacy or landmark")
        if not np.isfinite(communication_radius) or communication_radius <= 0:
            raise ValueError("communication_radius must be finite and positive")
        self.communication_mode, self.communication_radius = communication_mode, communication_radius
        self.base = simple_spread_v3.parallel_env(
            N=n, max_cycles=horizon, local_ratio=0.5, dynamic_rescaling=False)
        self.possible_agents = self.base.possible_agents[:]
        self.agents = []
        self.channel = None
        self.steps = 0

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Self pos/vel; K peers and K landmarks (dx, dy, mask); pooled inbox.
        extra = 5 * self.neighbors + 1 if self.communication_mode == "landmark" else 0
        return spaces.Box(-np.inf, np.inf, shape=(4 + 6 * self.neighbors + 6 + extra,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete([5, 2])  # movement, transmit gate

    @property
    def world(self):
        return self.base.unwrapped.world

    def reset(self, seed=None, options=None):
        self.base.reset(seed=seed, options=options)
        positions = spaced_landmarks(np.asarray([l.state.p_pos for l in self.world.landmarks]),
                                     self.base.unwrapped.np_random, self.min_landmark_distance, method=self.layout_sampler)
        for landmark, position in zip(self.world.landmarks, positions):
            landmark.state.p_pos[:] = position
        starts = spaced_agents(np.asarray([a.state.p_pos for a in self.world.agents]),
                               self.base.unwrapped.np_random, self.min_agent_distance, method=self.layout_sampler)
        for agent, position in zip(self.world.agents, starts):
            agent.state.p_pos[:] = position
        self.agents = self.possible_agents[:]
        self.steps = 0
        # Independent channel RNG avoids movement/action RNG coupling.
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.channel = (LandmarkChannel(self.n, self.loss, self.ttl, rng, self.communication_radius, self.neighbors)
                        if self.communication_mode == "landmark" else Channel(self.n, self.loss, self.ttl, rng))
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
        if self.communication_mode == "landmark":
            positions = np.asarray([a.state.p_pos for a in self.world.agents])
            landmarks = np.asarray([l.state.p_pos for l in self.world.landmarks])
            self.channel.refresh_seen(positions, landmarks, self.radius)
            inbox = self.channel.observe_all(positions)
            covering = self.channel.covering(positions, self.coverage_radius)
        for i, name in enumerate(self.possible_agents):
            agent = self.world.agents[i]
            observations[name] = np.concatenate([
                agent.state.p_pos, agent.state.p_vel,
                self._local_entities(agent.state.p_pos,
                                     [p for p in self.world.agents if p is not agent]),
                self._local_entities(agent.state.p_pos, self.world.landmarks),
                inbox[i] if self.communication_mode == "landmark" else self.channel.observe(i),
                *([np.asarray([covering[i]], np.float32)] if self.communication_mode == "landmark" else [])]).astype(np.float32)
        return observations

    def state(self):
        """Privileged centralized critic input; length depends on N."""
        physical = np.concatenate([*[np.concatenate([a.state.p_pos, a.state.p_vel])
                                 for a in self.world.agents],
                               *[l.state.p_pos for l in self.world.landmarks]]).astype(np.float32)
        if self.communication_mode == "landmark":
            positions = np.asarray([a.state.p_pos for a in self.world.agents])
            return np.concatenate([physical, self.channel.observe_all(positions).ravel()]).astype(np.float32)
        return physical

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
        traffic = (self.channel.advance(gates, np.asarray([a.state.p_pos for a in self.world.agents]),
                   np.asarray([a.state.p_vel for a in self.world.agents]),
                   np.asarray([l.state.p_pos for l in self.world.landmarks]), self.radius, self.coverage_radius)
                   if self.communication_mode == "landmark" else self.channel.advance(gates, payloads))
        _, _, terminated, truncated, _ = self.base.step({a: int(actions[a][0]) for a in active})
        self.steps += 1
        observations = self._observations()
        diagnostics = self.diagnostics()
        # Charge attempted transmissions including packets lost in transit; shared team objective.
        penalty = self.comm_cost * traffic.attempted_links / self.n
        team_task_reward = float(task_reward(
            diagnostics["mean_landmark_distance"] * self.n,
            diagnostics["collision_pairs"], diagnostics["success"], self.n))
        rewards = {a: team_task_reward - penalty for a in active}
        info = {**diagnostics, "task_reward": team_task_reward,
                "task_reward_version": TASK_REWARD_VERSION,
                "communication_penalty": penalty, **vars(traffic)}
        if self.communication_mode == "landmark":
            info.update(communication_protocol="landmark-radius-v1", communication_radius=self.communication_radius,
                        eligible_links=self.channel.last_eligible_links)
        if self.terminate_on_collision and diagnostics["collision_pairs"] > 0:
            terminated = {a: True for a in active}
            truncated = {a: False for a in active}
            info["success"] = False
            info["terminated_by_collision"] = True
            self.agents = []
        else:
            self.agents = self.base.agents[:]
        infos = {a: info.copy() for a in active}
        return observations, rewards, terminated, truncated, infos

    def close(self):
        self.base.close()
