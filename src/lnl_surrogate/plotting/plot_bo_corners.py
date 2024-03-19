from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult as SciPyOptimizeResult
from skopt.plots import plot_evaluations as skopt_plot_evaluations
from skopt.plots import plot_objective as skopt_plot_objective
from skopt.space import Space
from trieste.models import ProbabilisticModel
from trieste.space import SearchSpace


def _make_scipy_result(x, y, space: SearchSpace, model: ProbabilisticModel):
    """
    Create a SciPyOptimizeResult object from the given points and model.
    """
    bounds = Space(
        [
            (space.lower[i].numpy(), space.upper[i].numpy())
            for i in range(space.dimension.numpy())
        ]
    )
    min_idx = np.argmin(y)
    return SciPyOptimizeResult(
        dict(
            fun=y[min_idx],
            x=x[min_idx],
            success=True,
            func_vals=y,
            x_iters=x,
            models=[model],
            space=bounds,
        )
    )


def plot_model_partial_dependence(
    in_pts,
    out_pts,
    model,
    search_space: SearchSpace,
    truth: Dict = None,
    **plotting_kwargs
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, truths = _get_param_labels(truth)
    truths = truths if truths is not None else "result"
    res = _make_scipy_result(in_pts, out_pts, search_space, model)

    # increase these for higher resolution
    plotting_kwargs["n_points"] = plotting_kwargs.get("n_points", 25)
    plotting_kwargs["n_samples"] = plotting_kwargs.get("n_samples", 100)
    plotting_kwargs["levels"] = plotting_kwargs.get("levels", 5)

    ax = skopt_plot_objective(
        res,
        sample_source="result",
        dimensions=labels,
        minimum=truths,
        **plotting_kwargs
    )
    return _get_fig(ax)


def plot_evaluations(
    in_pts, out_pts, model, search_space: SearchSpace, truth: Dict = None
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, _ = _get_param_labels(truth)
    res = _make_scipy_result(in_pts, out_pts, search_space, model)
    ax = skopt_plot_evaluations(res, dimensions=labels)
    return _get_fig(ax)


def _get_param_labels(truths):
    labels, truths = None, None
    if isinstance(truths, dict):
        # get dict without lnl key
        truths = {k: v for k, v in truths.items() if k != "lnl"}
        labels = list(truths.keys())
        truths = list(truths.values())
    return labels, truths


def _get_fig(ax):
    if isinstance(ax, np.ndarray):
        fig = ax.flatten()[0].get_figure()
    else:
        fig = ax.get_figure()
    return fig
