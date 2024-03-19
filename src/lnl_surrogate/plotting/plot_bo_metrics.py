"""Makes a plot of the regret + distance of the surrogate model during the optimization."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


def _distances_between_consecutive_points(points: np.ndarray) -> np.ndarray:
    """Compute the distances between consecutive points."""
    dist = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[np.nan], dist])


def _min_point_per_iteration(points: np.ndarray) -> np.ndarray:
    """Compute the minimum point per iteration."""
    return np.minimum.accumulate(points)


def plot_bo_metrics(
    query_points: np.ndarray,
    objective_values: np.ndarray,
    color: str = "tab:blue",
    label: str = None,
    init_n_points: int = None,
    axes: plt.Axes = None,
    truth: float = None,
) -> plt.Figure:
    """Plot the regret and distance of the surrogate model during the optimization."""
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    fig = axes[0].get_figure()

    plot_convergence(
        objective_values, color, label, init_n_points, axes[0], truth
    )
    plot_distance(query_points, color, label, init_n_points, axes[1])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig


def plot_distance(
    points: np.ndarray,
    color: str = "tab:blue",
    label: str = None,
    init_n_points: int = None,
    ax: plt.Axes = None,
) -> None:
    """Plot the distance between consecutive points."""
    if ax is None:
        fig, ax = plt.subplots()

    distances = _distances_between_consecutive_points(points)
    n_calls = np.arange(len(points))
    if init_n_points:
        ax.axvline(
            init_n_points, color="gray", linestyle="--", label="Initial points"
        )

    ax.plot(n_calls, distances, color=color, label=label)
    ax.set_xlabel("Num $f(x)$ calls ($n$)")
    ax.set_ylabel("Distance between consecutive $x$")


def plot_convergence(
    points: np.ndarray,
    color: str = "tab:blue",
    label: str = None,
    init_n_points: int = None,
    ax: plt.Axes = None,
    true_minimum: float = None,
) -> None:
    """Plot the convergence of the surrogate model."""
    if ax is None:
        fig, ax = plt.subplots()

    min_points = _min_point_per_iteration(points)
    n_calls = np.arange(len(points))
    if init_n_points:
        ax.axvline(
            init_n_points,
            color="gray",
            linestyle="--",
            label="Initial points",
            zorder=-10,
        )
    if true_minimum:
        ax.axhline(
            true_minimum,
            color="red",
            linestyle="--",
            label="True minimum",
            zorder=-10,
        )

    ax.plot(n_calls, min_points, color=color, label=label)
    ax.set_xlabel("Num $f(x)$ calls ($n$)")
    ax.set_ylabel("min $f(x)$ after $n$ calls")
