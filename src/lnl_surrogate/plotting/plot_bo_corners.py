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
    Create a SciPyOptimizeResult object from the given y_pts and model.
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
    plotting_kwargs["n_points"] = plotting_kwargs.get("n_points", 50)
    plotting_kwargs["n_samples"] = plotting_kwargs.get("n_samples", 500)
    plotting_kwargs["levels"] = plotting_kwargs.get("levels", 5)

    ax = skopt_plot_objective(
        res,
        sample_source="random",
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
    labels, tru_vals = _get_param_labels(truth)
    # add truth to in_pts and out_pts
    # if truth:
    #     in_pts = np.vstack([in_pts, tru_vals])
    #     out_pts = np.append(out_pts, truth['lnl'])
    res = _make_scipy_result(in_pts, out_pts, search_space, model)
    ax = skopt_plot_evaluations(res, dimensions=labels)
    fig = _get_fig(ax)
    if truth:
        n_dims = in_pts.shape[1]
        # t_vals = np.array([tru_vals])
        # overplot_lines(fig, t_vals, color="tab:orange")
        # overplot_points(
        #     fig,
        #     [[np.nan if t is None else t for t in t_vals]],
        #     marker="s",
        #     color="tab:orange"
        # )
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:  # diagonal
                    if n_dims == 1:
                        ax_ = ax
                    else:
                        ax_ = ax[i, i]
                    ax_.vlines(
                        tru_vals[i], *ax_.get_ylim(), color="tab:orange"
                    )
                # lower triangle
                elif i > j:
                    ax_ = ax[i, j]
                    ax_.vlines(
                        tru_vals[j], *ax_.get_ylim(), color="tab:orange"
                    )
                    ax_.hlines(
                        tru_vals[i], *ax_.get_xlim(), color="tab:orange"
                    )
                    ax_.scatter(
                        tru_vals[j],
                        tru_vals[i],
                        c="tab:orange",
                        s=50,
                        lw=0.0,
                        marker="s",
                    )

    return fig


def _get_param_labels(truths):
    labels, truth_vals = None, None
    if isinstance(truths, dict):
        # get dict without lnl key
        truths = {k: v for k, v in truths.items() if k != "lnl"}
        labels = list(truths.keys())
        truth_vals = list(truths.values())
    return labels, truth_vals


def _get_fig(ax):
    if isinstance(ax, np.ndarray):
        fig = ax.flatten()[0].get_figure()
    else:
        fig = ax.get_figure()
    return fig
