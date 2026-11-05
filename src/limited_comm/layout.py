"""Rejection-sample agent/landmark layouts with Euclidean center-distance constraints."""
import numpy as np


def minimum_separation(positions):
    positions = np.asarray(positions)
    n = positions.shape[-2]
    distances = np.linalg.norm(positions[..., :, None, :] - positions[..., None, :, :], axis=-1)
    distances = np.where(np.eye(n, dtype=bool), np.inf, distances)
    return distances.min(axis=(-2, -1))


def spaced_positions(initial, rng, min_distance=0.0, entity="entity"):
    """Keep valid layouts; redraw whole invalid layouts uniformly in [-1,1]^2.

    Rejecting complete layouts avoids the order bias of placing points one by one.
    Accepts (N,2) or (B,N,2) arrays. This only samples reset positions.
    """
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError(f"min_{entity}_distance must be finite and nonnegative")
    result = np.asarray(initial, dtype=np.float64).copy()
    if min_distance == 0:
        return result
    if min_distance > np.sqrt(8) and result.shape[-2] > 1:
        raise ValueError(f"Requested {entity} separation is impossible in [-1,1]^2")
    batch = result.reshape(-1, result.shape[-2], 2)
    for _ in range(10_000):
        invalid = minimum_separation(batch) < min_distance
        if not invalid.any():
            return result
        batch[invalid] = rng.uniform(-1, 1, size=(int(invalid.sum()), batch.shape[1], 2))
    raise ValueError(f"Could not sample this {entity} separation; reduce spacing or entity count")


def spaced_landmarks(initial, rng, min_distance=0.0):
    """Sample static landmark centers; keep the historical public API."""
    return spaced_positions(initial, rng, min_distance, entity="landmark")


def spaced_agents(initial, rng, min_distance=0.4):
    """Sample starting agent centers at least min_distance apart radially."""
    return spaced_positions(initial, rng, min_distance, entity="agent")
