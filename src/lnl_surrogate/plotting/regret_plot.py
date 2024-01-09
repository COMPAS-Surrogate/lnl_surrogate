"""Makes a plot of the regret + distance of the surrogate model during the optimization."""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


def _compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def _distances_between_consecutive_points(points: np.ndarray) -> np.ndarray:
    """Compute the distances between consecutive points."""
    return np.ndarray([_compute_distance(points[i], points[i + 1]) for i in range(len(points) - 1)])


def _min_point_per_iteration(points: np.ndarray) -> np.ndarray:
    """Compute the minimum point per iteration."""
    return np.minimum.accumulate(points)

