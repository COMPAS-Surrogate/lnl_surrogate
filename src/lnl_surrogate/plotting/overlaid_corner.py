"""Help plot overlaid corners"""
import warnings
from typing import Dict, List

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# rcParams["font.size"] = 30
# rcParams["font.family"] = "serif"
# rcParams["font.sans-serif"] = ["Computer Modern Sans"]
# rcParams["text.usetex"] = True
# rcParams['axes.labelsize'] = 30
# rcParams['axes.titlesize'] = 30
# rcParams['axes.labelpad'] = 10
# rcParams['axes.linewidth'] = 2.5
# rcParams['axes.edgecolor'] = 'black'
# rcParams['xtick.labelsize'] = 25
# rcParams['xtick.major.size'] = 10.0
# rcParams['xtick.minor.size'] = 5.0
# rcParams['ytick.labelsize'] = 25
# rcParams['ytick.major.size'] = 10.0
# rcParams['ytick.minor.size'] = 5.0
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['xtick.minor.width'] = 1
# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.minor.width'] = 1
# plt.rcParams['ytick.major.width'] = 2.5
# plt.rcParams['xtick.top'] = True
# plt.rcParams['ytick.right'] = True

CORNER_KWARGS = dict(
    smooth=0.99,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=(0.16, 0.84),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


def plot_overlaid_corner(
    samples_list: List[pd.DataFrame],
    sample_labels: List[str],
    axis_labels: List[str] = None,
    colors: List[str] = None,
    fname: str = "corner.png",
    truths: Dict[str, float] = None,
    label: str = None,
):
    """Plots multiple corners on top of each other"""
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])
    # drop the 'log_likelhood' 'log_prior' columns from samples
    samples_list = [
        samples.drop(columns=["log_likelihood", "log_prior"])
        for samples in samples_list
    ]

    if colors is None:
        colors = [f"C{i}" for i in range(n)]

    if axis_labels is None:
        axis_labels = samples_list[0].columns

    CORNER_KWARGS.update(
        labels=axis_labels,
        ranges=get_axes_ranges(samples_list, truths),
        truths=[truths[col] for col in truths.keys()] if truths else None,
    )

    fig = corner.corner(
        samples_list[0].values,
        color=colors[0],
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx].values,
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=colors[idx],
            **CORNER_KWARGS,
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(1, ndim),
        loc="upper right",
    )
    # add textbox on top left corner with n_training_points
    fig.text(
        0.1,
        0.9,
        label,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )

    fig.savefig(fname)
    plt.close(fig)


def _get_data_ranges(data: pd.DataFrame) -> List[List[float]]:
    """Get the ranges of the data"""
    return [[data[col].min(), data[col].max()] for col in data.columns]


def get_axes_ranges(
    samples_list: List[pd.DataFrame],
    truths: Dict[str, float] = {},
    truth_thres=0.1,
) -> List[List[float]]:
    """Get the ranges of the data"""
    ranges = [_get_data_ranges(samples) for samples in samples_list]
    if truths:
        ranges.append(
            [
                [
                    truths[col] - truth_thres * truths[col],
                    truths[col] + truth_thres * truths[col],
                ]
                for col in truths.keys()
            ]
        )
    ranges = np.array(ranges)

    # set ax_range based on max and mins of all col
    ax_ranges = np.array(
        [
            [np.min(ranges[:, :, 0]), np.max(ranges[:, :, 1])]
            for _ in range(len(samples_list[0].columns))
        ]
    )

    return ax_ranges


def get_normalisation_weight(
    len_current_samples: int, len_of_longest_samples: int
):
    return np.ones(len_current_samples) * (
        len_of_longest_samples / len_current_samples
    )
