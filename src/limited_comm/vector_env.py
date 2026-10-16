"""Batched NumPy implementation of CommunicationSpread for CPU training.

Matches pinned MPE2 1.1.0 integration (position BEFORE velocity update).
Evaluation still uses the original PettingZoo adapter. Tests check parity.
"""
import numpy as np
from itertools import permutations
from .layout import spaced_landmarks


class VectorSpread:
    def __init__(self, batch=64, n=3, horizon=100, radius=0.6, neighbors=3,
                 ttl=5, loss=0.0, comm_cost=0.0, seed=42, min_landmark_distance=0.4):
        self.batch, self.n, self.horizon = batch, n, horizon
        self.radius, self.neighbors, self.ttl = radius, neighbors, ttl
        self.loss, self.comm_cost = loss, comm_cost
        self.min_landmark_distance = min_landmark_distance
        seeds = np.random.SeedSequence(seed).spawn(2)
        self.rng, self.channel_rng = [np.random.default_rng(s) for s in seeds]
        self.pos = np.zeros((batch, n, 2), dtype=np.float64)
        self.vel = np.zeros_like(self.pos)
        self.landmarks = np.zeros_like(self.pos)
        self.cache = np.zeros((batch, n, n, 4), dtype=np.float64)  # receiver, sender
        self.age = np.full((batch, n, n), ttl + 1, dtype=np.int64)
        self.diagonal = np.eye(n, dtype=bool)[None]
        self.steps = 0
        self.reset()

    def reset(self):
        self.pos[:] = self.rng.uniform(-1, 1, self.pos.shape)
        self.vel.fill(0)
        self.landmarks[:] = self.rng.uniform(-1, 1, self.landmarks.shape)
        self.landmarks[:] = spaced_landmarks(self.landmarks, self.rng, self.min_landmark_distance)
        self.cache.fill(0)
        self.age.fill(self.ttl + 1)
        self.steps = 0
        return self.observe(), self.state()

    def inbox(self):
        valid = self.age <= self.ttl
        count = valid.sum(axis=2)
        denom = np.maximum(count, 1)
        mean = (self.cache * valid[..., None]).sum(axis=2) / denom[..., None]
        age = (self.age * valid).sum(axis=2) / denom / self.ttl
        return np.concatenate([mean, (count / (self.n - 1))[..., None], age[..., None]], axis=-1)

    def _slots(self, relative, peer=False):
        distances = np.linalg.norm(relative, axis=-1)
        if peer:
            distances = np.where(self.diagonal, np.inf, distances)
        order = np.argsort(distances, axis=2, kind="stable")
        k = min(self.neighbors, self.n)
        order = order[:, :, :k]
        selected = np.take_along_axis(relative, order[..., None], axis=2)
        visible = np.take_along_axis(distances, order, axis=2) <= self.radius
        slots = np.zeros((self.batch, self.n, self.neighbors, 3))
        slots[:, :, :k, :2] = selected * visible[..., None]
        slots[:, :, :k, 2] = visible
        return slots.reshape(self.batch, self.n, -1)

    def observe(self):
        peers = self.pos[:, None, :, :] - self.pos[:, :, None, :]
        landmarks = self.landmarks[:, None, :, :] - self.pos[:, :, None, :]
        return np.concatenate([self.pos, self.vel, self._slots(peers, True),
                               self._slots(landmarks), self.inbox()], axis=-1).astype(np.float32)

    def state(self):
        return np.concatenate([np.concatenate([self.pos, self.vel], axis=-1).reshape(self.batch, -1),
                               self.landmarks.reshape(self.batch, -1),
                               self.inbox().reshape(self.batch, -1)], axis=-1).astype(np.float32)

    def diagnostics(self):
        distances = np.linalg.norm(self.landmarks[:, :, None, :] - self.pos[:, None, :, :], axis=-1)
        nearest = distances.min(axis=2)
        covered = nearest <= 0.1
        peers = np.linalg.norm(self.pos[:, :, None, :] - self.pos[:, None, :, :], axis=-1)
        pairs = np.triu(peers < 0.3, k=1).sum(axis=(1, 2))
        return {"coverage": covered.mean(axis=1), "success": covered.all(axis=1),
                "collision_pairs": pairs, "distance_sum": nearest.sum(axis=1)}

    def potential(self):
        """Negative optimal one-to-one assignment distance (three-agent training)."""
        distances = np.linalg.norm(self.pos[:, :, None, :] - self.landmarks[:, None, :, :], axis=-1)
        return -np.stack([distances[:, np.arange(self.n), order].sum(axis=1)
                          for order in permutations(range(self.n))], axis=1).min(axis=1)

    def visible_landmarks(self):
        distances = np.linalg.norm(self.landmarks[:, :, None, :] - self.pos[:, None, :, :], axis=-1)
        return distances.min(axis=2) <= self.radius

    def step(self, moves, gates):
        payload = np.concatenate([self.pos, self.vel], axis=-1)
        self.age = np.minimum(self.age + 1, self.ttl + 1)
        delivered = ((self.channel_rng.random((self.batch, self.n, self.n)) >= self.loss)
                     & ~self.diagonal & np.asarray(gates, dtype=bool)[:, None, :])
        self.cache = np.where(delivered[..., None], payload[:, None, :, :], self.cache)
        self.age = np.where(delivered, 1, self.age)

        delta = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        safe_distance = np.where(self.diagonal, 1.0, np.maximum(distance, 1e-12))
        penetration = np.logaddexp(0, -(distance - 0.3) / 0.001) * 0.001
        force = 100 * delta / safe_distance[..., None] * penetration[..., None]
        force = np.where(self.diagonal[..., None], 0, force).sum(axis=2)
        movement = np.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])[moves]
        self.pos += self.vel * 0.1
        self.vel = self.vel * 0.75 + (5 * movement + force) * 0.1
        self.steps += 1
        diagnostics = self.diagnostics()
        # local_ratio=0.5: half global distance reward, half mean local collision reward.
        reward = -0.5 * diagnostics["distance_sum"] - diagnostics["collision_pairs"] / self.n
        reward -= self.comm_cost * np.asarray(gates).sum(axis=1) * (self.n - 1) / self.n
        return self.observe(), self.state(), reward.astype(np.float32), self.steps >= self.horizon, diagnostics
