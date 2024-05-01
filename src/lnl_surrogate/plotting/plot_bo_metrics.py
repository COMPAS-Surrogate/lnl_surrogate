"""Makes a plot of the regret + distance of the surrogate model during the optimization."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from trieste.models import ProbabilisticModel


def _distances_between_consecutive_points(points: np.ndarray) -> np.ndarray:
    """Compute the distances between consecutive y_pts."""
    dist = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[np.nan], dist])


def _min_point_per_iteration(xpts: np.ndarray, ypts: np.ndarray) -> np.ndarray:
    """Compute the minimum point per iteration."""

    min_ys = np.minimum.accumulate(ypts)
    min_xs = xpts[np.argmin(ypts, axis=0)]

    return min_xs, min_ys


def plot_bo_metrics(
    query_points: np.ndarray,
    objective_values: np.ndarray,
    model: ProbabilisticModel,
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
        x_pts=query_points,
        y_pts=objective_values,
        model=model,
        color=color,
        init_n_points=init_n_points,
        ax=axes[0],
        true_minimum=truth,
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
    """Plot the distance between consecutive y_pts."""
    if ax is None:
        fig, ax = plt.subplots()

    distances = _distances_between_consecutive_points(points)
    n_calls = np.arange(len(points))
    if init_n_points:
        ax.axvline(
            init_n_points, color="gray", linestyle="--", label="Initial y_pts"
        )

    if len(points) > 30:
        # plot the last 30 y_pts
        n_calls = n_calls[-30:]
        distances = distances[-30:]

    ax.plot(n_calls, distances, color=color, label=label)
    ax.set_xlabel("Num $f(x)$ calls ($n$)")
    ax.set_ylabel("Distance between consecutive $x$")


def plot_convergence(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    model: ProbabilisticModel,
    color: str = "tab:blue",
    init_n_points: int = None,
    ax: plt.Axes = None,
    true_minimum: float = None,
    n_points_to_plot: int = 30,
) -> None:
    """Plot the convergence of the surrogate model."""
    if ax is None:
        fig, ax = plt.subplots()

    n_calls = np.arange(len(y_pts))
    x_mins, y_mins = _min_point_per_iteration(x_pts, y_pts)
    y_gp, y_gpunc = model.predict(x_pts)
    y_gp_upper = y_gp + y_gpunc
    y_gp_lower = y_gp - y_gpunc

    if n_points_to_plot > 0 and len(y_pts) > n_points_to_plot:
        # plot the last n_points_to_plot
        n_calls = n_calls[-n_points_to_plot:]
        y_mins = y_mins[-n_points_to_plot:]
        x_mins = x_mins[-n_points_to_plot:]
        y_pts = y_pts[-n_points_to_plot:]
        y_gp_upper = y_gp_upper[-n_points_to_plot:]
        y_gp_lower = y_gp_lower[-n_points_to_plot:]

    if init_n_points:
        ax.axvline(
            init_n_points,
            color="gray",
            linestyle="--",
            label="Initial y_pts",
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

    ax.scatter(n_calls, y_pts, color=color, label="$f(x)$")
    ax.plot(n_calls, y_mins, color=color, label="$f(x)_{\\rm min}$")
    ax.fill_between(
        n_calls,
        y_gp_upper.numpy().flatten(),
        y_gp_lower.numpy().flatten(),
        color="tab:orange",
        alpha=0.3,
        label="GP 1$\sigma$",
    )
    ax.set_xlabel("Num $f(x)$ calls ($n$)")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc="upper right")
