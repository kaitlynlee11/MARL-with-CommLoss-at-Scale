"""Shared task reward for training, evaluation, and replay accounting."""
import numpy as np


TASK_REWARD_VERSION = "coverage-gated-distance-v1"


def task_reward(distance_sum, collision_pairs, full_coverage, n):
    """Stop distance costs while fully covered; always retain collision costs.

    This is a per-state rule, not an absorbing success: losing coverage resumes
    the distance cost, and reaching coverage does not end the episode.
    """
    return -0.5 * np.where(full_coverage, 0.0, distance_sum) - np.asarray(collision_pairs) / n
