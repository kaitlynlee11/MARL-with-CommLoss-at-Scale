"""Rejection-sample agent/landmark layouts with Euclidean center-distance constraints."""
import numpy as np


def minimum_separation(positions):
    positions = np.asarray(positions)
    n = positions.shape[-2]
    distances = np.linalg.norm(positions[..., :, None, :] - positions[..., None, :, :], axis=-1)
    distances = np.where(np.eye(n, dtype=bool), np.inf, distances)
    return distances.min(axis=(-2, -1))


def spaced_positions(initial, rng, min_distance=0.0, entity="entity", method="whole"):
    """Keep valid layouts; redraw whole invalid layouts uniformly in [-1,1]^2.

    Rejecting complete layouts avoids the order bias of placing points one by one.
    Accepts (N,2) or (B,N,2) arrays. This only samples reset positions.
    """
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError(f"min_{entity}_distance must be finite and nonnegative")
    if method not in ("whole", "sequential"):
        raise ValueError("layout sampler must be whole or sequential")
    result = np.asarray(initial, dtype=np.float64).copy()
    if min_distance == 0:
        return result
    if min_distance > np.sqrt(8) and result.shape[-2] > 1:
        raise ValueError(f"Requested {entity} separation is impossible in [-1,1]^2")
    batch = result.reshape(-1, result.shape[-2], 2)
    if method == "sequential":
        # Whole-configuration rejection becomes impractical for dense N=20
        # layouts. Sequential rejection preserves the same hard constraints and
        # bounds, but has a different spatial distribution; callers must label it.
        for layout in batch:
            for _ in range(64):
                placed = 0
                for index in range(len(layout)):
                    for _ in range(32):
                        candidates = rng.uniform(-1, 1, size=(256, 2))
                        valid = (np.linalg.norm(candidates[:, None, :] - layout[None, :index, :], axis=-1)
                                 >= min_distance).all(axis=1)
                        if valid.any():
                            layout[index] = candidates[np.flatnonzero(valid)[0]]
                            placed += 1
                            break
                    else:
                        break
                if placed == len(layout):
                    layout[:] = layout[rng.permutation(len(layout))]
                    break
            else:
                raise ValueError(f"Could not sequentially sample {entity} layout at requested spacing")
        return result
    for _ in range(10_000):
        invalid = minimum_separation(batch) < min_distance
        if not invalid.any():
            return result
        batch[invalid] = rng.uniform(-1, 1, size=(int(invalid.sum()), batch.shape[1], 2))
    raise ValueError(f"Could not sample this {entity} separation; reduce spacing or entity count")


def spaced_landmarks(initial, rng, min_distance=0.0, method="whole"):
    """Sample static landmark centers; keep the historical public API."""
    return spaced_positions(initial, rng, min_distance, entity="landmark", method=method)


def spaced_agents(initial, rng, min_distance=0.4, method="whole"):
    """Sample starting agent centers at least min_distance apart radially."""
    return spaced_positions(initial, rng, min_distance, entity="agent", method=method)
