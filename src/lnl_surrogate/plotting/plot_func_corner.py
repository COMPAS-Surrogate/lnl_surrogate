import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import corner

from typing import Callable
from trieste.space import SearchSpace
from trieste.models import ProbabilisticModel
from scipy.optimize import OptimizeResult as SciPyOptimizeResult


#
#
# def _scatter_and_density_map(*args, **kwargs):
#     x, y = args[:2]
#     search_space = kwargs['search_space']
#     model = kwargs['model']
#     sample_pts
#
#     range = kwargs.get('range', None)
#     ax = kwargs['ax']
#
#
#
#     ax.scatter(x, y, zorder=-1, color='k', m='x')
#     corner.core._set_xlim(kwargs['new_fig'], ax, range[0])
#     corner.core._set_ylim(kwargs['new_fig'], ax, range[1])
#     if range is None:
#         range = [[x.min(), x.max()], [y.min(), y.max()]]
#
#
def model_1d_marginal(space:SearchSpace, model:ProbabilisticModel, i:int, samples:np.ndarray, n_points=40):
    """
    Calculate the model 1D marginal output.

    This uses the given model to calculate the model-output
    for all the samples, where the given dimension is fixed at
    regular intervals between its bounds.

    This shows how the given dimension affects the model-output
    when the influence of all other dimensions are marginalised out.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The dimension for which to calculate the maringal output.

    samples : np.array, space=(n_points, n_dims)
        Randomly sampled and transformed points to use when integrating
        the model function at each of the `n_points` when using marginal

    n_points : int, default=40
        Number of points at which to evaluate the marginal dependence
        along each dimension `i`.

    Returns
    -------
    xi : np.array
        The model input points at which the marginalisation was evaluated.

    yi : np.array
        The marginalised model output at each point `xi`.

    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and marginalising either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    def _calc(x):
        """
        Helper-function to calculate the marginal output
        value for the given model, when setting
        the index'th dimension of the search-space to the value x,
        and then marginalising over all samples.
        """
        rvs_ = np.array(samples)  # copy
        # We replace the values in the dimension that we want to keep
        # fixed
        rvs_[:, dim_locs[i]:dim_locs[i + 1]] = x
        # In case of `x_eval=None` rvs conists of random samples.
        # Calculating the mean of these samples is how partial dependence
        # is implemented.
        return np.mean(model.predict(rvs_))

    xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
    # Calculate the partial dependence for all the points.
    yi = [_calc(x) for x in xi_transformed]

    return xi, yi
#
#
#
# def model_2d_marginal(space:SearchSpace, model:ProbabilisticModel, i:int, j:int, samples:np.ndarray, n_points=40):
#     """
#     Calculate the partial dependence for two dimensions in the search-space.
#
#     This uses the given model to calculate the marginal objective value
#     for all the samples, where the given dimensions are fixed at
#     regular intervals between their bounds.
#
#     This shows how the given dimensions affect the objective value
#     when the influence of all other dimensions are marginalized out.
#
#     Parameters
#     ----------
#     space : `Space`
#         The parameter space over which the minimization was performed.
#
#     model
#         Surrogate model for the objective function.
#
#     i : int
#         The first dimension for which to calculate the partial dependence.
#
#     j : int
#         The second dimension for which to calculate the partial dependence.
#
#     samples : np.array, space=(n_points, n_dims)
#         Randomly sampled and transformed points to use when marginalising
#         the model function at each of the `n_points` when using partial
#         dependence.
#
#     n_points : int, default=40
#         Number of points at which to evaluate the partial dependence
#         along each dimension `i` and `j`.
#
#     Returns
#     -------
#     xi : np.array, space=n_points
#         The points at which the partial dependence was evaluated.
#
#     yi : np.array, space=n_points
#         The points at which the partial dependence was evaluated.
#
#     zi : np.array, space=(n_points, n_points)
#         The marginal value of the objective function at each point `(xi, yi)`.
#     """
#     # The idea is to step through one dimension, evaluating the model with
#     # that dimension fixed and marginalising either over random values or over
#     # the given ones in x_val in all other dimensions.
#     # (Or step through 2 dimensions when i and j are given.)
#     # Categorical dimensions make this interesting, because they are one-
#     # hot-encoded, so there is a one-to-many mapping of input dimensions
#     # to transformed (model) dimensions.
#
#     # dim_locs[i] is the (column index of the) start of dim i in
#     # sample_points.
#     # This is usefull when we are using one hot encoding, i.e using
#     # categorical values
#     dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])
#
#     def _calc(x, y):
#         """
#         Helper-function to calculate the marginal predicted
#         objective value for the given model, when setting
#         the index1'th dimension of the search-space to the value x
#         and setting the index2'th dimension to the value y,
#         and then marginalising over all samples.
#         """
#         rvs_ = np.array(samples)  # copy
#         rvs_[:, dim_locs[j]:dim_locs[j + 1]] = x
#         rvs_[:, dim_locs[i]:dim_locs[i + 1]] = y
#         return np.mean(model.predict(rvs_))
#
#     xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
#     yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)
#     # Calculate the partial dependence for all combinations of these points.
#     zi = [[_calc(x, y) for x in xi_transformed] for y in yi_transformed]
#
#     # Convert list-of-list to a numpy array.
#     zi = np.array(zi)
#
#     return xi, yi, zi




def _evenly_sample(space: SearchSpace, n_points):
    """Return n_points^Dimensions (evenly spaced points) from a search-space"""
    gridpts = int(np.power(n_points, 1/space.dimension.numpy()))
    meshgrid_arrays = [np.linspace(lower, upper, num=gridpts) for lower, upper in
                       zip(space.lower, space.upper)]
    meshgrid = np.array(np.meshgrid(*meshgrid_arrays))
    # unroll the meshgrid into a list of points
    return np.array([x.flatten() for x in meshgrid]).T


def rejection_sample(samples, weights, num_samples_to_generate):
    indices = np.random.choice(len(samples), size=num_samples_to_generate, p=weights / np.sum(weights))
    return samples[indices], weights[indices]


def plot_func_corner(pts, predict_fn, search_space: SearchSpace, grid_points=50000, **kwargs) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    _grid_pts = _evenly_sample(search_space, 100)

    # y_mean, y_var = predict_fn(_grid_pts)
    # fig = plt.tricontourf(_grid_pts[:, 0], _grid_pts[:, 1], y_mean.flatten(), 100)
    # plt.show()

    _grid_pts = search_space.sample(grid_points).numpy()
    y_model, y_var = predict_fn(_grid_pts)
    y_model *= -1
    y_model = y_model - y_model.min()
    y_model = y_model / y_model.max()


    _grid_pts, y_model = rejection_sample(_grid_pts, y_model, int(len(_grid_pts)*0.9))



    fig = corner.corner(
        data=_grid_pts,
        weights=y_model,
        label_kwargs=dict(fontsize=30),
        title_kwargs=dict(fontsize=16),
        truth_color="tab:orange",
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,
        contourf_kwargs=dict(cmap="viridis"),
        max_n_ticks=3,
        verbose=False,
        use_math_text=True,
        data_kwargs=dict(
            alpha=0.75,
            ms=1,
        ),
    )
    #
    # corner.overplot_points(
    #     fig,
    #     [[np.nan if t is None else t for t in pts]],
    #     marker="x",
    #     color='black',
    # )

    return fig






def plot_hack_func_corner(pts, predict_fn, search_space: SearchSpace, grid_points=50000, **kwargs) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    _grid_pts = _evenly_sample(search_space, 100)

    # y_mean, y_var = predict_fn(_grid_pts)
    # fig = plt.tricontourf(_grid_pts[:, 0], _grid_pts[:, 1], y_mean.flatten(), 100)
    # plt.show()

    _grid_pts = search_space.sample(grid_points).numpy()
    y_model, y_var = predict_fn(_grid_pts)
    y_model *= -1
    y_model = y_model - y_model.min()
    y_model = y_model / y_model.max()


    _grid_pts, y_model = rejection_sample(_grid_pts, y_model, int(len(_grid_pts)*0.9))



    fig = corner.corner(
        data=_grid_pts,
        weights=y_model,
        label_kwargs=dict(fontsize=30),
        title_kwargs=dict(fontsize=16),
        truth_color="tab:orange",
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,
        contourf_kwargs=dict(cmap="viridis"),
        max_n_ticks=3,
        verbose=False,
        use_math_text=True,
        data_kwargs=dict(
            alpha=0.75,
            ms=1,
        ),
    )
    #
    # corner.overplot_points(
    #     fig,
    #     [[np.nan if t is None else t for t in pts]],
    #     marker="x",
    #     color='black',
    # )

    return fig
