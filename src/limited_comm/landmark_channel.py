"""Exploratory radius-limited landmark/coverage packets for both simulators.

Only directly observed landmarks enter a sender's episode memory. Transmit gates
remain external/learned. Cached coverage is a timestamped report, never live truth.
"""
import numpy as np

from .channel import Traffic

PROTOCOL = "landmark-radius-v1"
PROTOCOL_STATUS = "exploratory"
HEADER_BYTES = 28  # position/velocity, covering flag, record count, send step
LANDMARK_BYTES = 16  # int32 landmark ID, float32 x/y, float32 covering flag


class VectorLandmarkChannel:
    def __init__(self, batch, n, loss, ttl, rng, communication_radius=1., neighbors=3):
        if batch < 1 or n < 2 or ttl < 1 or not 0 <= loss <= 1 or neighbors < 1:
            raise ValueError("Invalid landmark channel dimensions, loss, or TTL")
        if not np.isfinite(communication_radius) or communication_radius <= 0:
            raise ValueError("Communication radius must be finite and positive")
        self.batch, self.n, self.loss, self.ttl = batch, n, loss, ttl
        self.rng, self.communication_radius, self.neighbors = rng, communication_radius, neighbors
        self.known = np.zeros((batch, n, n), bool)  # world, sender, landmark
        self.known_xy = np.zeros((batch, n, n, 2), np.float64)
        self.kinematics = np.zeros((batch, n, n, 4), np.float32)  # world, receiver, sender
        self.age = np.full((batch, n, n), ttl + 1, int)
        self.received_known = np.zeros((batch, n, n, n), bool)
        self.received_xy = np.zeros((batch, n, n, n, 2), np.float32)
        self.received_covering = np.zeros((batch, n, n, n), bool)
        self.sender_covering = np.zeros((batch, n, n), bool)
        self.sent_step = np.full((batch, n, n), -1, int)
        self.clock = np.zeros(batch, int)

    @property
    def inbox_dim(self):
        return 6 + 5 * self.neighbors

    def reset(self, mask=None):
        ids = np.arange(self.batch) if mask is None else np.flatnonzero(mask)
        for array in (self.known, self.known_xy, self.kinematics, self.received_known,
                      self.received_xy, self.received_covering, self.sender_covering):
            array[ids] = 0
        self.age[ids] = self.ttl + 1
        self.sent_step[ids] = -1
        self.clock[ids] = 0

    def refresh_seen(self, positions, landmarks, sensing_radius):
        distances = np.linalg.norm(landmarks[:, None] - positions[:, :, None], axis=-1)
        order = np.argsort(distances, axis=-1, kind="stable")[:, :, :self.neighbors]
        visible = np.zeros_like(self.known)
        np.put_along_axis(visible, order, np.take_along_axis(distances, order, axis=-1) <= sensing_radius, axis=-1)
        self.known_xy = np.where(visible[..., None], landmarks[:, None], self.known_xy)
        self.known |= visible

    def covering(self, positions, coverage_radius=.1):
        return self.known & (np.linalg.norm(self.known_xy - positions[:, :, None], axis=-1) <= coverage_radius)

    def advance(self, gates, positions, velocities, landmarks, sensing_radius=.6, coverage_radius=.1):
        gates = np.asarray(gates)
        if gates.shape != (self.batch, self.n) or not np.isin(gates, [0, 1]).all():
            raise ValueError("One binary transmit gate is required for each agent/world")
        for array in (positions, velocities, landmarks):
            if np.shape(array) != (self.batch, self.n, 2) or not np.isfinite(array).all():
                raise ValueError("World positions and velocities must be finite (batch, n, 2) arrays")
        self.refresh_seen(positions, landmarks, sensing_radius)
        covering = self.covering(positions, coverage_radius)
        self.age = np.minimum(self.age + 1, self.ttl + 1)
        distances = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
        eligible = (distances <= self.communication_radius) & ~np.eye(self.n, dtype=bool)[None]
        attempted = eligible & gates[:, None, :].astype(bool)
        # Draws are indexed sender, receiver, including out-of-range/silent links.
        draws = self.rng.random((self.batch, self.n, self.n)).transpose(0, 2, 1)
        delivered = attempted & (draws >= self.loss)
        payload = np.concatenate([positions, velocities], axis=-1).astype(np.float32)
        self.kinematics = np.where(delivered[..., None], payload[:, None], self.kinematics)
        self.received_known = np.where(delivered[..., None], self.known[:, None], self.received_known)
        self.received_xy = np.where(delivered[..., None, None], self.known_xy[:, None], self.received_xy).astype(np.float32)
        self.received_covering = np.where(delivered[..., None], covering[:, None], self.received_covering)
        self.sender_covering = np.where(delivered, covering.any(axis=-1)[:, None], self.sender_covering)
        self.sent_step = np.where(delivered, self.clock[:, None, None], self.sent_step)
        self.age = np.where(delivered, 1, self.age)
        sizes = HEADER_BYTES + LANDMARK_BYTES * self.known.sum(axis=-1)
        self.clock += 1
        return {"broadcasts": gates.sum(axis=1).astype(int), "eligible_links": eligible.sum(axis=(1, 2)),
                "attempted_links": attempted.sum(axis=(1, 2)), "delivered_links": delivered.sum(axis=(1, 2)),
                "payload_bytes": (attempted * sizes[:, None]).sum(axis=(1, 2))}

    def observe(self, positions):
        valid = self.age <= self.ttl
        count = valid.sum(axis=2)
        denom = np.maximum(count, 1)
        mean = (self.kinematics * valid[..., None]).sum(axis=2) / denom[..., None]
        age = (self.age * valid).sum(axis=2) / denom / self.ttl
        pooled = np.concatenate([mean, (count / (self.n - 1))[..., None], age[..., None]], axis=-1)
        reported = self.received_known & valid[..., None]
        ages = np.where(reported, self.age[..., None], self.ttl + 1)
        newest = ages.min(axis=2)
        senders = ages.argmin(axis=2)
        coords = np.take_along_axis(self.received_xy, senders[:, :, None, :, None], axis=2).squeeze(2)
        # A sender reporting "I am not covering" cannot invalidate another
        # sender's coverage report. Only that sender's update/expiry clears it.
        positive = self.received_covering & reported
        occupied = positive.any(axis=2)
        status_age = np.where(occupied, np.where(positive, ages, self.ttl + 1).min(axis=2), newest)
        known = newest <= self.ttl
        relative = coords - positions[:, :, None]
        distance = np.where(known, np.linalg.norm(relative, axis=-1), np.inf)
        k = min(self.neighbors, self.n)
        order = np.argsort(distance, axis=-1, kind="stable")[:, :, :k]
        seen = np.take_along_axis(known, order, axis=2)
        slots = np.zeros((self.batch, self.n, self.neighbors, 5), np.float32)
        slots[:, :, :k, :2] = np.take_along_axis(relative, order[..., None], axis=2) * seen[..., None]
        slots[:, :, :k, 2] = seen
        slots[:, :, :k, 3] = np.take_along_axis(occupied, order, axis=2) * seen
        slots[:, :, :k, 4] = np.take_along_axis(status_age, order, axis=2) / self.ttl * seen
        return np.concatenate([pooled, slots.reshape(self.batch, self.n, -1)], axis=-1).astype(np.float32)


class LandmarkChannel:
    """Single-world adapter with inspectable decoded packet records."""
    def __init__(self, n, loss, ttl, rng, communication_radius=1., neighbors=3):
        self.impl = VectorLandmarkChannel(1, n, loss, ttl, rng, communication_radius, neighbors)
        self.n, self.ttl = n, ttl
        self.last_eligible_links = 0

    def refresh_seen(self, positions, landmarks, sensing_radius):
        self.impl.refresh_seen(np.asarray(positions)[None], np.asarray(landmarks)[None], sensing_radius)

    def advance(self, gates, positions, velocities, landmarks, sensing_radius=.6, coverage_radius=.1):
        info = self.impl.advance(np.asarray(gates)[None], np.asarray(positions)[None], np.asarray(velocities)[None],
                                 np.asarray(landmarks)[None], sensing_radius, coverage_radius)
        self.last_eligible_links = int(info["eligible_links"][0])
        return Traffic(**{key: int(info[key][0]) for key in Traffic.__dataclass_fields__})

    def observe_all(self, positions):
        return self.impl.observe(np.asarray(positions)[None])[0]

    def covering(self, positions, coverage_radius=.1):
        return self.impl.covering(np.asarray(positions)[None], coverage_radius)[0].any(axis=-1)

    def packet(self, receiver, sender):
        channel = self.impl
        if channel.age[0, receiver, sender] > self.ttl:
            return None
        records = []
        for landmark in np.flatnonzero(channel.received_known[0, receiver, sender]):
            records.append({"landmark_id": int(landmark),
                            "position": channel.received_xy[0, receiver, sender, landmark].tolist(),
                            "sender_covering": bool(channel.received_covering[0, receiver, sender, landmark])})
        return {"sender_position_velocity": channel.kinematics[0, receiver, sender].tolist(),
                "sender_covering_any": bool(channel.sender_covering[0, receiver, sender]),
                "sent_step": int(channel.sent_step[0, receiver, sender]),
                "age_steps": int(channel.age[0, receiver, sender]), "landmarks": records}

    @property
    def cache(self):
        return {(r, s): (self.packet(r, s), int(self.impl.age[0, r, s]))
                for r in range(self.n) for s in range(self.n) if self.impl.age[0, r, s] <= self.ttl}
