"""Rejection-sample landmark layouts with a Euclidean center-distance constraint."""
import numpy as np


def minimum_separation(positions):
    positions = np.asarray(positions)
    n = positions.shape[-2]
    distances = np.linalg.norm(positions[..., :, None, :] - positions[..., None, :, :], axis=-1)
    distances = np.where(np.eye(n, dtype=bool), np.inf, distances)
    return distances.min(axis=(-2, -1))


def spaced_landmarks(initial, rng, min_distance=0.0):
    """Keep valid layouts; redraw whole invalid layouts uniformly in [-1,1]^2.

    Rejecting complete layouts avoids the order bias of placing points one by one.
    Accepts (N,2) or (B,N,2) arrays. Landmarks are static after reset.
    """
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_landmark_distance must be finite and nonnegative")
    result = np.asarray(initial, dtype=np.float64).copy()
    if min_distance == 0:
        return result
    if min_distance > np.sqrt(8) and result.shape[-2] > 1:
        raise ValueError("Requested landmark separation is impossible in [-1,1]^2")
    batch = result.reshape(-1, result.shape[-2], 2)
    for _ in range(10_000):
        invalid = minimum_separation(batch) < min_distance
        if not invalid.any():
            return result
        batch[invalid] = rng.uniform(-1, 1, size=(int(invalid.sum()), batch.shape[1], 2))
    raise ValueError("Could not sample this landmark separation; reduce spacing or landmark count")
