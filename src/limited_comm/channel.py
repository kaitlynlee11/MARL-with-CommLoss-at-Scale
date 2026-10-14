"""Seeded lossy all-to-all unicast channel with one-step delivery."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Traffic:
    broadcasts: int = 0
    attempted_links: int = 0
    delivered_links: int = 0
    payload_bytes: int = 0


class Channel:
    def __init__(self, n: int, loss: float, ttl: int, rng: np.random.Generator):
        if n < 2 or not 0 <= loss <= 1 or ttl < 1:
            raise ValueError("Require n >= 2, loss in [0, 1], and ttl >= 1")
        self.n, self.loss, self.ttl, self.rng = n, loss, ttl, rng
        self.cache: dict[tuple[int, int], tuple[np.ndarray, int]] = {}

    def advance(self, gates: np.ndarray, payloads: np.ndarray) -> Traffic:
        """Payload at t becomes visible at t+1; failed delivery retains old data."""
        gates = np.asarray(gates)
        payloads = np.asarray(payloads, dtype=np.float32)
        if gates.shape != (self.n,) or not np.isin(gates, [0, 1]).all():
            raise ValueError("One binary gate is required per agent")
        if payloads.shape != (self.n, 4) or not np.isfinite(payloads).all():
            raise ValueError("Each payload must contain four finite float32 values")
        self.cache = {key: (data, age + 1) for key, (data, age) in self.cache.items()
                      if age + 1 <= self.ttl}
        # Draw every link each step, even for silent senders, to pair loss RNG across schedules.
        draws = self.rng.random((self.n, self.n))
        delivered = 0
        for sender in range(self.n):
            if gates[sender]:
                for receiver in range(self.n):
                    if receiver != sender and draws[sender, receiver] >= self.loss:
                        self.cache[receiver, sender] = (payloads[sender].copy(), 1)
                        delivered += 1
        broadcasts = int(gates.sum())
        attempts = broadcasts * (self.n - 1)
        return Traffic(broadcasts, attempts, delivered, attempts * 4 * 4)

    def observe(self, receiver: int) -> np.ndarray:
        """Mean payload, fraction of senders present, and normalized mean age."""
        messages = [value for (r, _), value in self.cache.items() if r == receiver]
        if not messages:
            return np.zeros(6, dtype=np.float32)
        return np.asarray([*np.mean([data for data, _ in messages], axis=0),
                           len(messages) / (self.n - 1),
                           np.mean([age for _, age in messages]) / self.ttl], dtype=np.float32)
