import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import corner


def _scatter_and_color_by_order(*args, **kwargs):
    x, y = args[:2]
    range = kwargs.get('range', None)
    ax = kwargs['ax']
    if range is None:
        range = [[x.min(), x.max()], [y.min(), y.max()]]
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(x)))
    ax.scatter(x, y, zorder=-1, c=colors)
    corner.core._set_xlim(kwargs['new_fig'], ax, range[0])
    corner.core._set_ylim(kwargs['new_fig'], ax, range[1])


def plot_evaluation_corner(pts:np.ndarray)->plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    corner.core.hist2d = _scatter_and_color_by_order
    fig = corner.corner(
        data=pts,
        label_kwargs=dict(fontsize=30),
        title_kwargs=dict(fontsize=16),
        truth_color="tab:orange",
        plot_density=False,
        plot_datapoints=True,
        plot_contours=False,
        fill_contours=False,
        max_n_ticks=3,
        verbose=False,
        use_math_text=True,
        data_kwargs=dict(
            alpha=0.75,
            ms=1,
        ),
    )
    return fig
